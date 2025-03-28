Here's a comparison of key Ollama LLMs based on available data as of March 2025:

| Model             | Parameters | Storage Size | Context Window | Tool Use | Function Calling | Time consumed | Quality |
|-------------------|------------|--------------|----------------|----------|------------------|---------------|---------|
| llama3.2          | 3B         | 2.0GB        | 130K tokens    | Y        | Y                | ~3-5s         | ++      |
| llama3.3          | 70B        | 43GB         | 128K tokens    | Y        | Y                | 45s           | ++      |
| phi4-mini         | 3.8B       | 2.5GB        | 128K tokens    | Y        | Y                | ~3-5s         | +       |
| mistral-small     | 24B        | 14GB         | 128K tokens    | Y        | Y                | 10-15s        | ++      |
| mistral-nemo      | 12B        | 7.1GB        | 128K tokens    | Y        | Y                | 5s            | -       |
| deepseek-r1:1.5b  | 1.5B       | 1.1GB        | 128K tokens    | Y        | Y*               | ~3-5s         | +       |
| deepseek-r1:7b    | 7B         | 4.7GB        | 128K tokens    | Y        | Y*               | ~3-5s         | +      |
| deepseek-r1:70B   | 70B        | 43GB         | 128K tokens    | Y        | Y*               | 60s           | ++      |
| gemma3:27b        | 27B        | 17GB         | 128K tokens    | Y        | Y                | 20s           | +       |
| qwq               | Unknown    | 20GB         | 131K tokens    | N        | N                | 40s           | +++     |


*DeepSeek-R1 function calling capabilities are currently unstable according to official documentation