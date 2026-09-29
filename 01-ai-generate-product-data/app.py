"""Streamlit app for generating fictional e-commerce product data."""

import json
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


GEMINI_MODEL_ID = "gemini-3.1-flash-lite"
GROQ_MODEL_ID = "openai/gpt-oss-20b"
REQUIRED_FIELDS = {
    "product_id",
    "sku",
    "product_name",
    "category",
    "brand",
    "short_description",
    "price",
    "currency",
    "stock_quantity",
    "tags",
}


class ProductValidationError(ValueError):
    """Raised when the model response is not a valid product list."""


class Product(BaseModel):
    """Strict Pydantic model for a valid product record."""

    model_config = ConfigDict(extra="forbid")

    product_id: str
    sku: str
    product_name: str
    category: str
    brand: str
    short_description: str
    price: float
    currency: str
    stock_quantity: int
    tags: list[str]

    @field_validator("product_id", "sku", "product_name", "category", "brand", "short_description", "currency")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Field must be a non-empty string.")
        return value.strip()

    @field_validator("price")
    @classmethod
    def validate_price(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Price must be numeric.")
        if value <= 0:
            raise ValueError("Price must be greater than 0.")
        return float(value)

    @field_validator("stock_quantity")
    @classmethod
    def validate_stock(cls, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Stock quantity must be an integer.")
        if value < 0:
            raise ValueError("Stock quantity must be greater than or equal to 0.")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        if not isinstance(value, list) or not value or any(not isinstance(tag, str) or not tag.strip() for tag in value):
            raise ValueError("Tags must be a non-empty list of strings.")
        return [tag.strip() for tag in value]


def diagnose_generation_error(error: Exception, provider: str = "Gemini", api_key: str = "") -> Tuple[str, str]:
    """Return a helpful, API-key-safe message and technical detail for the UI."""
    detail = " ".join(str(error).split())
    if api_key:
        detail = detail.replace(api_key, "[REDACTED]")
    detail = re.sub(r"Bearer\s+[^\s,;]+", "Bearer [REDACTED]", detail, flags=re.IGNORECASE)
    detail = re.sub(r"(x-goog-api-key[=:]\s*)[^\s,;]+", r"\1[REDACTED]", detail, flags=re.IGNORECASE)
    detail = detail[:800] or error.__class__.__name__
    lower_detail = detail.lower()

    if "no api keys configured" in lower_detail or "gemini_api_key" in lower_detail or "groq_api_key" in lower_detail:
        return "Add GEMINI_API_KEY and GROQ_API_KEY to .env before generating products.", detail
    if "both gemini and groq failed" in lower_detail or "gemini and groq" in lower_detail:
        return "Gemini and Groq are unavailable or their free-tier limits were reached. Please try again later.", detail

    if provider == "Gemini":
        if "api_key_invalid" in lower_detail or "api key not valid" in lower_detail or "401" in lower_detail:
            return "Gemini rejected the API key. Check GEMINI_API_KEY in .env and restart the app.", detail
        if "permission_denied" in lower_detail or "403" in lower_detail or "forbidden" in lower_detail:
            return "Your Gemini API key does not have permission to use the selected model.", detail
        if "not_found" in lower_detail or ("model" in lower_detail and "not found" in lower_detail):
            return f"Gemini could not find the configured model ({GEMINI_MODEL_ID}).", detail
        if "429" in lower_detail or "resource_exhausted" in lower_detail or "rate limit" in lower_detail or "quota" in lower_detail:
            return "Gemini free-tier quota or rate limits were reached. Retrying with Groq automatically.", detail
        if "timeout" in lower_detail or "timed out" in lower_detail:
            return "The Gemini request timed out. Retrying with Groq automatically.", detail
        return "Gemini request failed. Retrying with Groq automatically.", detail

    if provider == "Groq":
        if "401" in lower_detail or "unauthorized" in lower_detail or "api key" in lower_detail and "invalid" in lower_detail:
            return "Groq rejected the API key. Check GROQ_API_KEY in .env and restart the app.", detail
        if "403" in lower_detail or "forbidden" in lower_detail:
            return "Your Groq API key does not have permission to use the selected model.", detail
        if "429" in lower_detail or "rate limit" in lower_detail or "quota" in lower_detail or "exhausted" in lower_detail:
            return "Groq free-tier quota or rate limits were reached. Please try again later.", detail
        return "Groq request failed. Please try again later.", detail

    return "Unable to generate products. Open Technical details below to diagnose the request.", detail


def build_prompt() -> str:
    """Return instructions that make the expected response unambiguous."""
    return """
Generate exactly 5 diverse, entirely fictional e-commerce products.
Return ONLY a valid JSON array. Do not include Markdown, explanations, or code fences.

Each array item must use exactly these keys:
product_id, sku, product_name, category, brand, short_description,
price, currency, stock_quantity, tags.

Rules:
- product_id and sku must be unique strings.
- price must be a non-negative number.
- currency must be a three-letter uppercase code, such as USD.
- stock_quantity must be a non-negative whole number.
- tags must be a JSON array of one or more strings.
- Make product names, categories, brands, and descriptions varied.
- Do not use real brands or real products.

Example shape (do not reuse these values):
[{"product_id":"PROD-001","sku":"SKU-001","product_name":"...",
"category":"...","brand":"...","short_description":"...","price":19.99,
"currency":"USD","stock_quantity":12,"tags":["..."]}]
""".strip()


def extract_json_array(response_text: str) -> List[Dict[str, Any]]:
    """Decode an array even if a model accidentally adds surrounding text."""
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)

    decoder = json.JSONDecoder()
    start = cleaned.find("[")
    if start == -1:
        raise ProductValidationError("The model did not return a JSON array.")

    try:
        data, _ = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as error:
        raise ProductValidationError("The model returned malformed JSON.") from error

    if not isinstance(data, list):
        raise ProductValidationError("The response must be a JSON array.")
    return data


def validate_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate required Python-level product fields and normalize values."""
    if not isinstance(products, list):
        raise ProductValidationError("The response must be a JSON array of products.")
    if len(products) != 5:
        raise ProductValidationError("The response must contain exactly five products.")

    normalized: List[Dict[str, Any]] = []
    product_ids = set()
    skus = set()
    errors: List[str] = []

    for index, product in enumerate(products, start=1):
        if not isinstance(product, dict):
            errors.append(f"Product {index} is not an object.")
            continue

        missing = sorted(REQUIRED_FIELDS - set(product.keys()))
        if missing:
            errors.append(f"Product {index} is missing required fields: {', '.join(missing)}.")
            continue

        text_fields = [
            "product_id", "sku", "product_name", "category", "brand", "short_description",
        ]
        for field in text_fields:
            value = product.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Product {index} field '{field}' is missing or empty.")

        if isinstance(product.get("price"), bool) or not isinstance(product.get("price"), (int, float)):
            errors.append(f"Product {index} price is not numeric.")
        elif float(product["price"]) <= 0:
            errors.append(f"Product {index} price must be greater than 0.")

        if isinstance(product.get("stock_quantity"), bool) or not isinstance(product.get("stock_quantity"), int):
            errors.append(f"Product {index} stock quantity must be an integer.")
        elif product["stock_quantity"] < 0:
            errors.append(f"Product {index} stock quantity must be greater than or equal to 0.")

        currency = product.get("currency")
        if not isinstance(currency, str) or not currency.strip():
            errors.append(f"Product {index} currency is missing or empty.")

        tags = product.get("tags")
        if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            errors.append(f"Product {index} tags are missing or invalid.")

        product_id = str(product.get("product_id", "")).strip()
        sku = str(product.get("sku", "")).strip()
        if product_id and product_id in product_ids:
            errors.append(f"Product {index} product_id must be unique.")
        if sku and sku in skus:
            errors.append(f"Product {index} sku must be unique.")
        if product_id:
            product_ids.add(product_id)
        if sku:
            skus.add(sku)

        if not errors or not any(error.startswith(f"Product {index}") for error in errors):
            normalized_product = dict(product)
            normalized_product["product_id"] = product_id
            normalized_product["sku"] = sku
            normalized_product["price"] = round(float(product["price"]), 2)
            normalized_product["tags"] = [tag.strip() for tag in tags] if isinstance(tags, list) else []
            normalized.append(normalized_product)

    if errors:
        raise ProductValidationError("; ".join(errors))

    return normalized


def validate_products_pydantic(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply strict Pydantic validation to each generated product."""
    validated_products: List[Dict[str, Any]] = []
    errors: List[str] = []

    for index, product in enumerate(products, start=1):
        try:
            model = Product(**product)
            validated_products.append(model.model_dump())
        except ValidationError as error:
            errors.append(f"Product {index}: {error.errors()[0]['msg'] if error.errors() else 'invalid data'}.")

    if errors:
        raise ProductValidationError("; ".join(errors))

    return validated_products


def generate_with_gemini(api_key: str) -> List[Dict[str, Any]]:
    """Call the Gemini API and validate the returned product list."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=build_prompt(),
        config={
            "response_mime_type": "application/json",
            "temperature": 0.9,
            "max_output_tokens": 1500,
        },
    )
    response_text = response.text
    if not isinstance(response_text, str) or not response_text.strip():
        raise ProductValidationError("The model returned an empty response.")
    return validate_products(extract_json_array(response_text))


def generate_with_groq(api_key: str) -> List[Dict[str, Any]]:
    """Call the Groq API and validate the returned product list."""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL_ID,
            "messages": [{"role": "user", "content": build_prompt()}],
            "temperature": 0.9,
            "max_tokens": 1500,
        },
        timeout=60,
    )

    if response.status_code == 429:
        raise RuntimeError("Groq free-tier quota was reached or rate-limited.")
    if response.status_code in {401, 403}:
        raise RuntimeError(f"Groq authentication failed: {response.text[:200]}")
    response.raise_for_status()

    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise ProductValidationError("Groq returned no choices.")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProductValidationError("Groq returned an empty response.")

    return validate_products(extract_json_array(content))


def generate_products() -> List[Dict[str, Any]]:
    """Try Gemini first and, on failure, retry once with Groq if configured."""
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    providers: List[Tuple[str, str, Any]] = []
    if gemini_key:
        providers.append(("Gemini", gemini_key, generate_with_gemini))
    if groq_key:
        providers.append(("Groq", groq_key, generate_with_groq))

    if not providers:
        raise RuntimeError("No API keys configured. Add GEMINI_API_KEY and GROQ_API_KEY in .env.")

    last_error: Exception | None = None

    for provider_name, api_key, provider_function in providers:
        try:
            products = provider_function(api_key)
            python_valid_products = validate_products(products)
            pydantic_valid_products = validate_products_pydantic(python_valid_products)
            return pydantic_valid_products
        except Exception as error:
            last_error = error
            if provider_name == "Gemini" and groq_key:
                continue
            break

    if last_error is not None:
        raise RuntimeError("Both Gemini and Groq failed to generate products. Please try again later.") from last_error
    raise RuntimeError("Unable to generate products.")


def main() -> None:
    st.set_page_config(page_title="AI E-commerce Product Generator", page_icon="🛍️")
    st.title("Generate fictional product records with AI")
    st.write("Create five fictional product records for demos, development, and testing")

    if st.button("Generate 5 Products", type="primary"):
        try:
            with st.spinner("Generating fictional products with AI..."):
                products = generate_products()

            st.success("Python validation passed.")
            st.success("Pydantic validation passed.")
            st.session_state["products"] = pd.DataFrame(products)
        except ProductValidationError as error:
            st.error(f"Validation failed: {error}")
        except Exception as error:
            load_dotenv()
            message, detail = diagnose_generation_error(
                error,
                provider="Groq" if "groq" in str(error).lower() or "both gemini and groq" in str(error).lower() else "Gemini",
                api_key=os.getenv("GROQ_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
            )
            st.error("AI generation failed. Please try again later.")
            with st.expander("Technical details"):
                st.code(f"{error.__class__.__name__}: {detail}", language="text")

    products_df = st.session_state.get("products")
    if products_df is not None:
        st.dataframe(products_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download CSV",
            data=products_df.to_csv(index=False).encode("utf-8"),
            file_name="fictional_products.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
