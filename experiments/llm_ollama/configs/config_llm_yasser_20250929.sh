model: "gpt-oss:120b"  # gpt-oss:20b  llama3.1:8b  gpt-oss:120b  deepseek-r1:7b  llama3.1:latest 
options:
  temperature: 0.2
  top_p: 0.9
  num_ctx: 4096
  num_predict: 256
io:
  input_path: "data/input/gyneco_texts.csv"     # CSV avec une colonne 'text'
  input_text_col: "text"
  output_dir: "data/output"
  prompt_path: "experiments/llm_ollama/prompts/symptoms.prompt.txt"
meta:
  task: "Predire symptomes"
  dataset_name: "gyneco_texts"
