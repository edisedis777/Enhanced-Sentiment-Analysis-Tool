import datetime
import os
import sys
import time
import argparse
import csv
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from colorama import Fore, Style, init
from tqdm import tqdm

# Handle import errors gracefully
try:
    from transformers import pipeline, logging
    import torch
    # Suppress non-critical warnings
    logging.set_verbosity_error()
except ImportError as e:
    print(f"Error: Required library not found - {e}")
    print("Please install required packages with: pip install transformers torch tqdm colorama matplotlib")
    sys.exit(1)

class SentimentAnalyzer:
    """A class for sentiment analysis with enhanced features"""
    
    # Sentiment label mapping
    LABEL_MAP = {
        'LABEL_0': {'name': 'Negative', 'color': Fore.RED},
        'LABEL_1': {'name': 'Neutral', 'color': Fore.YELLOW},
        'LABEL_2': {'name': 'Positive', 'color': Fore.GREEN}
    }
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment", 
                 log_file: str = "sentiment_log.csv", 
                 device: Optional[str] = None) -> None:
        """Initialize the sentiment analyzer with specified model and settings"""
        self.model_name = model_name
        self.log_file = log_file
        
        # Determine device (CPU or GPU) - use CUDA if available
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize colorama for colored terminal output
        init(autoreset=True)
        
        # Load model
        self._load_model()
        
        # Initialize log file with header if it doesn't exist
        self._init_log_file()
        
        # Track statistics for this session
        self.stats = {'Positive': 0, 'Neutral': 0, 'Negative': 0, 'total': 0}
    
    def _load_model(self) -> None:
        """Load the sentiment analysis model with error handling"""
        try:
            print(f"Loading sentiment analysis model on {self.device}...")
            start_time = time.time()
            self.classifier = pipeline(
                'sentiment-analysis', 
                model=self.model_name, 
                device=0 if self.device == "cuda" else -1
            )
            load_time = time.time() - start_time
            print(f"{Fore.GREEN}Model loaded successfully in {load_time:.2f} seconds on {self.device.upper()}.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {e}{Style.RESET_ALL}")
            print("Falling back to default model...")
            try:
                self.classifier = pipeline('sentiment-analysis')
                print(f"{Fore.YELLOW}Default model loaded instead.{Style.RESET_ALL}")
            except Exception as e2:
                print(f"{Fore.RED}Fatal error: Could not load any model: {e2}{Style.RESET_ALL}")
                sys.exit(1)
    
    def _init_log_file(self) -> None:
        """Initialize the log file with a header if it doesn't exist"""
        if not os.path.exists(self.log_file):
            try:
                with open(self.log_file, "w", newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Timestamp", "Text", "Sentiment", "Confidence", "ProcessingTime"])
            except IOError as e:
                print(f"{Fore.RED}Warning: Could not create log file: {e}{Style.RESET_ALL}")
                self.log_file = None
    
    def analyze(self, text: str) -> Dict:
        """Analyze the sentiment of a text and return the result"""
        if not text.strip():
            return {"error": "Empty text provided"}
        
        try:
            start_time = time.time()
            result = self.classifier(text[:512])  # Truncate to model limit
            processing_time = time.time() - start_time
            
            label = result[0]['label']
            score = result[0]['score']
            sentiment = self.LABEL_MAP.get(label, {'name': 'Unknown', 'color': Fore.WHITE})['name']
            
            # Update statistics
            self.stats[sentiment] += 1
            self.stats['total'] += 1
            
            # Log to file
            self._log_result(text, sentiment, score, processing_time)
            
            return {
                "sentiment": sentiment,
                "score": score,
                "processing_time": processing_time
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts in batch mode for efficiency"""
        results = []
        
        for text in tqdm(texts, desc="Analyzing batch", unit="text"):
            results.append(self.analyze(text))
            
        return results
    
    def _log_result(self, text: str, sentiment: str, score: float, processing_time: float) -> None:
        """Log the analysis result to the CSV file"""
        if self.log_file is None:
            return
            
        try:
            with open(self.log_file, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    text[:100] + ("..." if len(text) > 100 else ""),  # Truncate long texts in log
                    sentiment,
                    f"{score:.4f}",
                    f"{processing_time:.4f}"
                ])
        except IOError:
            pass  # Silent fail on logging errors
    
    def display_result(self, result: Dict) -> None:
        """Display the analysis result to the user with color formatting"""
        if "error" in result:
            print(f"{Fore.RED}Error during analysis: {result['error']}{Style.RESET_ALL}")
            return
            
        sentiment = result["sentiment"]
        score = result["score"]
        processing_time = result.get("processing_time", 0)
        
        # Get color for this sentiment
        color = next((item['color'] for item in self.LABEL_MAP.values() 
                     if item['name'] == sentiment), Fore.WHITE)
        
        # Display with color coding and confidence bands
        confidence_bar = self._get_confidence_bar(score)
        
        print(f"\nSentiment: {color}{sentiment}{Style.RESET_ALL}")
        print(f"Confidence: {confidence_bar} {score:.2f}")
        print(f"Processing time: {processing_time:.4f} seconds\n")
    
    def _get_confidence_bar(self, score: float, width: int = 20) -> str:
        """Generate a visual confidence bar"""
        filled = int(score * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar
    
    def show_statistics(self) -> None:
        """Display statistics about the current session"""
        if self.stats['total'] == 0:
            print("No analyses performed yet.")
            return
            
        print(f"\n{Fore.CYAN}Session Statistics:{Style.RESET_ALL}")
        print(f"Total analyses: {self.stats['total']}")
        
        for sentiment, count in sorted(
            {k: v for k, v in self.stats.items() if k != 'total'}.items(),
            key=lambda x: x[1], reverse=True
        ):
            color = next((item['color'] for item in self.LABEL_MAP.values() 
                     if item['name'] == sentiment), Fore.WHITE)
            percentage = (count / self.stats['total']) * 100 if self.stats['total'] > 0 else 0
            bar_length = int(percentage / 5)  # 20 chars = 100%
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            print(f"{color}{sentiment}: {count} ({percentage:.1f}%) {bar}{Style.RESET_ALL}")
    
    def visualize_statistics(self) -> None:
        """Generate a pie chart of sentiment distribution"""
        if self.stats['total'] == 0:
            print("No data to visualize.")
            return
            
        # Filter out the 'total' key and get sentiment counts
        sentiment_data = {k: v for k, v in self.stats.items() if k != 'total' and v > 0}
        
        if not sentiment_data:
            print("No sentiment data to visualize.")
            return
            
        try:
            # Define colors for the pie chart
            colors = ['#FF6B6B', '#FFD166', '#06D6A0']  # red, yellow, green
            
            # Create the pie chart
            plt.figure(figsize=(8, 6))
            plt.pie(
                sentiment_data.values(),
                labels=sentiment_data.keys(),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                shadow=True
            )
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            plt.title('Sentiment Distribution', fontsize=16)
            
            # Save and show
            plt.savefig('sentiment_distribution.png')
            plt.show()
            print(f"Chart saved as 'sentiment_distribution.png'")
        except Exception as e:
            print(f"Error creating visualization: {e}")
            print("Matplotlib may not be properly configured.")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Sentiment Analysis Tool')
    parser.add_argument('--model', type=str, default="cardiffnlp/twitter-roberta-base-sentiment",
                        help='Model to use for sentiment analysis')
    parser.add_argument('--file', type=str, help='Analyze texts from a file, one per line')
    parser.add_argument('--output', type=str, help='Output results to a specified file')
    parser.add_argument('--no-log', action='store_true', help='Disable logging of results')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    
    args = parser.parse_args()
    
    # Configure analyzer
    device = "cpu" if args.cpu else None
    log_file = None if args.no_log else "sentiment_log.csv"
    
    # Create analyzer instance
    analyzer = SentimentAnalyzer(model_name=args.model, log_file=log_file, device=device)
    
    # Check for file input mode
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
                
            print(f"Analyzing {len(texts)} texts from '{args.file}'...")
            results = analyzer.analyze_batch(texts)
            
            # Write results to output file if specified
            if args.output:
                with open(args.output, 'w', newline='', encoding='utf-8') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(["Text", "Sentiment", "Confidence"])
                    
                    for text, result in zip(texts, results):
                        if "error" not in result:
                            writer.writerow([
                                text[:100] + ("..." if len(text) > 100 else ""),
                                result["sentiment"],
                                f"{result['score']:.4f}"
                            ])
                print(f"Results written to '{args.output}'")
            
            # Show summary statistics
            analyzer.show_statistics()
            analyzer.visualize_statistics()
            return
        except FileNotFoundError:
            print(f"Error: File '{args.file}' not found.")
            return
        except Exception as e:
            print(f"Error processing file: {e}")
            return
    
    # Interactive mode
    print(f"{Fore.CYAN}╔══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
    print(f"{Fore.CYAN}║              Enhanced Sentiment Analysis Tool            ║{Style.RESET_ALL}")
    print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
    print(f"✓ Enter text to analyze its sentiment")
    print(f"✓ Commands: 'stats' (show statistics), 'viz' (visualize results), 'quit' (exit)")
    print(f"✓ Results are logged to {log_file or 'memory only (logging disabled)'}")
    
    while True:
        try:
            # Get user input
            text = input("\n> ")
            
            # Check for commands
            if text.lower() == 'quit':
                break
            elif text.lower() == 'stats':
                analyzer.show_statistics()
                continue
            elif text.lower() == 'viz':
                analyzer.visualize_statistics()
                continue
            elif not text.strip():
                print("Please enter some text.")
                continue
                
            # Analyze sentiment
            result = analyzer.analyze(text)
            analyzer.display_result(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Show final statistics before exit
    print("\nFinal Statistics:")
    analyzer.show_statistics()
    print(f"\nThank you for using the Enhanced Sentiment Analysis Tool!")
    

if __name__ == "__main__":
    main()
