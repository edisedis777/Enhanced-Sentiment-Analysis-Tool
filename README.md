# Enhanced Sentiment Analysis Tool
A powerful and user-friendly command-line tool for analyzing the sentiment of text using state-of-the-art transformer models.

![Screenshot 2025-03-15 at 16 32 59](https://github.com/user-attachments/assets/a1a84f8a-ecda-426a-a382-9eddac4f4f47)


## Features

- 🧠 **Powerful NLP**: Leverages Hugging Face transformer models for accurate sentiment analysis
- 🚀 **High Performance**: Automatic GPU acceleration when available
- 📊 **Data Visualization**: Generate charts and statistics from your analyses
- 🌈 **Rich Output**: Color-coded results with visual confidence indicators
- 📝 **Comprehensive Logging**: Track all analyses in a structured CSV format
- 📦 **Batch Processing**: Analyze multiple texts from files for efficiency
- 🛠️ **Flexible Usage**: Interactive mode or command-line options

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-tool.git
   cd sentiment-analysis-tool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install them directly:
   ```bash
   pip install transformers torch tqdm colorama matplotlib
   ```

## Usage

### Interactive Mode
Simply run the script without arguments for interactive mode:

```bash
python sentiment_analyzer.py
```

Enter text at the prompt to analyze sentiment. Special commands:
- `stats`: Display statistics about your session
- `viz`: Generate and display a visualization of sentiment distribution
- `quit`: Exit the program

### Command-Line Arguments
```bash
python sentiment_analyzer.py --file input.txt --output results.csv --model cardiffnlp/twitter-roberta-base-sentiment
```

Available options:
- `--model`: Specify the Hugging Face model to use (default: "cardiffnlp/twitter-roberta-base-sentiment")
- `--file`: Process multiple texts from a file (one per line)
- `--output`: Save results to a specified CSV file
- `--no-log`: Disable automatic logging of results
- `--cpu`: Force CPU usage even if GPU is available

## Examples

### Analyzing a Single Text
```bash
python sentiment_analyzer.py
> I absolutely love this new feature, it works perfectly!
```

### Batch Processing From File
```bash
python sentiment_analyzer.py --file customer_reviews.txt --output sentiment_results.csv
```

### Using a Different Model
```bash
python sentiment_analyzer.py --model distilbert-base-uncased-finetuned-sst-2-english
```

## Advanced Usage

### Integration in Python Scripts
You can import and use the `SentimentAnalyzer` class in your own Python code:

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze("I'm really enjoying this new software!")
print(f"Sentiment: {result['sentiment']}, Score: {result['score']}")
```

### Processing Multiple Texts Efficiently
```python
texts = ["Great product!", "Not satisfied with the quality", "It's okay, but not amazing"]
results = analyzer.analyze_batch(texts)
```

## Logging and Data Analysis
By default, all analyses are logged to `sentiment_log.csv` with timestamps, which allows for:

- Tracking sentiment trends over time
- Building datasets for further analysis
- Generating comprehensive reports

## Visualization
The tool can generate pie charts showing the distribution of sentiments in your analyzed texts:

![sentiment_distribution](https://github.com/user-attachments/assets/772f4dd7-609c-4295-a4e6-0db060f00d2f)

## Dependencies
- [transformers](https://github.com/huggingface/transformers): State-of-the-art NLP models
- [torch](https://pytorch.org/): Deep learning framework
- [tqdm](https://github.com/tqdm/tqdm): Progress bars for batch processing
- [colorama](https://github.com/tartley/colorama): Terminal text coloring
- [matplotlib](https://matplotlib.org/): Data visualization

## Contributing
Contributions are welcome!

## Acknowledgments
- [Hugging Face](https://huggingface.co/) for their excellent transformers library
- [Cardiff NLP](https://github.com/cardiffnlp) for the pre-trained sentiment analysis model

## License
Distributed under the GNU Affero General Public License v3.0 License. See `LICENSE` for more information.
