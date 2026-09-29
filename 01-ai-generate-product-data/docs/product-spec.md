# AI E-commerce Product Generator

## Problem

Developers, testers, and product teams often need realistic product data for demos, application development, and QA validation. Creating that data manually is time-consuming and inconsistent.

## Goal

Build a simple AI-powered application that generates five fictional e-commerce product records with a reliable validation pipeline and automatic provider fallback.

## Target Users

- Product managers
- Developers
- QA engineers
- Demo and testing teams

## Current Scope

The application provides a single-click flow:

1. The user clicks the "Generate 5 Products" button.
2. The app requests five fictional product records from an AI provider.
3. The generated response is parsed as JSON.
4. The data is validated using custom Python logic and a strict Pydantic model.
5. The results are displayed in a table.
6. The user can download the results as CSV.

## Product Schema

Each generated product contains the following fields:

- `product_id`
- `sku`
- `product_name`
- `category`
- `brand`
- `short_description`
- `price`
- `currency`
- `stock_quantity`
- `tags`

## Functional Requirements

1. Display a "Generate 5 Products" button.
2. Generate exactly five fictional products.
3. Return the data as a valid JSON array.
4. Validate output before rendering.
5. Display the products in a Streamlit table.
6. Allow the user to download the result as CSV.
7. Retry using Groq if Gemini fails or rate limits are reached.

## AI and Provider Requirements

- Use Gemini as the primary AI provider.
- Model in use: `gemini-3.1-flash-lite`
- If Gemini fails, automatically retry once with Groq.
- Fallback model in use: `openai/gpt-oss-20b`
- Generated data must be entirely fictional and suitable for demos, development, and testing.
- Output must be structured and consistent enough for validation.

## Validation Requirements

The application validates products in two layers:

1. Python validation checks for required fields, uniqueness, numeric constraints, and tag format.
2. Pydantic validation using a strict model to enforce schema rules.

Required validation constraints include:

- exactly five products must be returned
- all string fields must be non-empty
- `price` must be numeric and greater than 0
- `stock_quantity` must be an integer and greater than or equal to 0
- `currency` must be present
- `tags` must be a non-empty array of strings
- `product_id` and `sku` must be unique

## Error Handling Requirements

- Missing or invalid configuration must produce a clear setup message.
- API failures must be handled safely.
- API keys must not be exposed in the UI.
- If the first retry using Groq succeeds, the user is not shown an error.
- If both providers fail, the app shows a generic AI generation failure message.

## Non-Functional Requirements

- Simple and easy-to-use UI
- Minimal configuration with `.env` keys
- Graceful handling of provider rate limits and quota errors
- Secure handling of API keys
- CSV export for downstream use and demos

## Future Enhancements

- Select the number of products
- Select a product category or theme
- Add product quality scoring
- Improve duplicate detection and validation coverage
- Add additional product attributes or richer product descriptions
