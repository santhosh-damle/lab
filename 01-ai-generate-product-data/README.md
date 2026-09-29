# AI E-commerce Product Generator

A lightweight Streamlit app that generates five fictional e-commerce product records. The app uses Gemini first and automatically falls back to Groq if the primary request fails or hits a quota or rate-limit condition.

## Features

- Generate exactly five fictional product records with one click
- AI-based product generation with Gemini as the primary provider
- Automatic fallback to Groq when Gemini fails or is unavailable
- A single fallback retry is used for generation failures
- Custom Python validation before rendering results
- Strict Pydantic validation for schema and field constraints
- Display results in a table and allow CSV export
- User-facing error handling without exposing API keys

## Tech Stack

- Python
- Streamlit
- Pandas
- Pydantic
- Google GenAI SDK (`google.genai`)
- Requests for Groq API calls
- python-dotenv for environment variables

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys to `.env`:

   ```env
   GEMINI_API_KEY=
   GROQ_API_KEY=
   ```

4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Validation Rules

The app validates each generated product using Python checks before display and then applies a strict Pydantic model:

- `product_id` is required and unique
- `sku` is required and unique
- `product_name`, `category`, `brand`, and `short_description` are required
- `price` is numeric and greater than 0
- `currency` is required
- `stock_quantity` is an integer and greater than or equal to 0
- `tags` is a non-empty list of strings
- the response must contain exactly five products

## Error Handling

- Gemini is attempted first.
- If Gemini fails, the app retries once using Groq.
- If Groq also fails or is unavailable, the user sees a clear AI generation error message.
- If the fallback succeeds, no error message is shown.

## Model Configuration

The current app configuration is:

- Gemini model: `gemini-3.1-flash-lite`
- Groq model: `openai/gpt-oss-20b`

## Screenshots

### App initialization

![App initialized](docs/images/1%20Initialized.png)

### Generation in progress

![Generation in progress](docs/images/1%20Generation%20in%20progress.png)

### Generation complete

![Generation complete](docs/images/3%20Generation%20complete.png)

### CSV download

![CSV downloaded](docs/images/4%20CSV%20downloaded.png)

## Notes

- The `.env` file is ignored by Git.
- API keys are redacted in technical error details.
- All generated products are fictional and intended for demos, development, and testing.
