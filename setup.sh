mkdir -p ~/.streamlit/
printf '[server]\nport = %s\nheadless = true\n' "$PORT" > ~/.streamlit/config.toml
