# Reddit Community Toxicity Detection & Social Network Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview

A comprehensive machine learning and social network analysis project that detects and analyzes toxic behavior patterns across 40 Reddit communities using Natural Language Processing (NLP) and graph-based community detection algorithms.

This project combines state-of-the-art transformer models with network science to understand how toxicity spreads and manifests in online communities.

---

## 🎯 Key Highlights

- 📊 **Dataset**: Analyzed **1 million Reddit comments** from **40 diverse subreddits**
- 🤖 **AI Model**: BERT-based toxicity detection using **unitary/toxic-bert**
- 🕸️ **Network Analysis**: Built interaction network with **40 nodes** and **134 edges**
- 📈 **Network Metrics**: Density of **0.172**, Average degree of **6.70**
- 🎨 **Visualizations**: Network graphs, correlation heatmaps, toxicity distributions

---

## 🏆 Key Findings

### Top 10 Most Toxic Subreddits (by Average Toxicity Score)

| Rank | Subreddit | Toxicity Score |
|------|-----------|----------------|
| 1 | r/gonewild | 0.347 |
| 2 | r/RoastMe | 0.318 |
| 3 | r/AmItheAsshole | 0.314 |
| 4 | r/wallstreetbets | 0.279 |
| 5 | r/ChapoTrapHouse | 0.278 |
| 6 | r/trashy | 0.272 |
| 7 | r/aww | 0.221 |
| 8 | r/unpopularopinion | 0.208 |
| 9 | r/relationship_advice | 0.207 |
| 10 | r/Animemes | 0.207 |

### Network Statistics
- **Total Communities Analyzed**: 40
- **Network Connections**: 134 edges
- **Network Density**: 0.172 (moderate connectivity)
- **Average Node Degree**: 6.70 connections per subreddit
- **Clustering Coefficient**: Indicates community structure
- **Modularity**: Detected distinct community clusters

---

## 🛠️ Technologies & Tools

### Machine Learning & NLP
- **Transformers**: BERT-based model (Hugging Face)
- **PyTorch**: Deep learning framework
- **scikit-learn**: ML utilities and metrics

### Network Analysis
- **NetworkX**: Graph creation and analysis
- **python-louvain**: Community detection algorithm

### Data Processing & Visualization
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical graphics

### Development Environment
- **Jupyter Notebook**: Interactive development
- **Google Colab**: Cloud computing platform

---

## 📁 Project Structure

```
reddit-toxicity-detection/
│
├── SNA_Research.ipynb          # Main analysis notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
│
├── data/
│   ├── raw/                    # Original dataset (not included)
│   ├── processed/              # Cleaned data
│   └── samples/                # Sample data for demo
│
├── results/
│   ├── figures/                # Generated plots
│   │   ├── network_graph.png
│   │   ├── toxicity_distribution.png
│   │   └── correlation_heatmap.png
│   └── metrics/                # Performance metrics
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or Google Colab
- 8GB+ RAM recommended
- (Optional) GPU for faster processing

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Jashan-Sood/reddit-toxicity-detection.git
cd reddit-toxicity-detection
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
```bash
# Option 1: Using kagglehub (in notebook)
import kagglehub
path = kagglehub.dataset_download("smagnan/1-million-reddit-comments-from-40-subreddits")

# Option 2: Manual download from Kaggle
# https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits
```

---

## 💻 Usage

### Running the Notebook

1. **Launch Jupyter Notebook**
```bash
jupyter notebook SNA_Research.ipynb
```

2. **Or use Google Colab**
   - Upload `SNA_Research.ipynb` to Google Colab
   - Import dataset using kagglehub
   - Run all cells

### Step-by-Step Analysis

The notebook is organized into the following sections:

1. **Data Loading & Preprocessing**
   - Import 1M Reddit comments
   - Clean and filter data
   - Sample 2,000 comments for analysis

2. **Toxicity Detection**
   - Load pre-trained BERT model
   - Generate toxicity scores
   - Classify comments

3. **Exploratory Data Analysis**
   - Statistical summaries
   - Distribution plots
   - Correlation analysis

4. **Network Construction**
   - Build subreddit interaction network
   - Calculate network metrics
   - Visualize network structure

5. **Community Detection**
   - Apply Louvain clustering algorithm
   - Identify community groups
   - Analyze community characteristics

6. **Results & Insights**
   - Generate visualizations
   - Export findings
   - Statistical testing

---

## 📊 Methodology

### 1. Data Collection
- **Source**: Kaggle dataset with 1M Reddit comments
- **Coverage**: 40 diverse subreddits
- **Timeframe**: Historical Reddit data
- **Sampling**: Random sample of 2,000 comments

### 2. Toxicity Detection Pipeline
```
Raw Comments → Text Preprocessing → BERT Tokenization → 
Toxicity Classification → Score Aggregation → Analysis
```

**Model Details:**
- **Model**: `unitary/toxic-bert`
- **Architecture**: BERT transformer (110M parameters)
- **Training**: Pre-trained on toxic comment dataset
- **Output**: Toxicity probability (0-1 scale)

### 3. Network Analysis
- **Graph Type**: Undirected weighted network
- **Nodes**: Subreddits (40 communities)
- **Edges**: Interaction strength between subreddits
- **Algorithm**: Louvain method for community detection
- **Metrics**: Degree centrality, betweenness, clustering coefficient

### 4. Statistical Analysis
- Correlation analysis between toxicity and engagement
- Distribution analysis of toxicity scores
- Community-level aggregation
- Hypothesis testing

---

## 📈 Results & Insights

### Network Characteristics
✅ **Moderate Connectivity**: Network density of 0.172 indicates communities are somewhat connected but not fully interconnected

✅ **Hub Communities**: Some subreddits act as bridges connecting different community clusters

✅ **Community Structure**: Louvain algorithm detected distinct community groups with shared characteristics

### Toxicity Patterns
📍 **High Variance**: Toxicity scores range from 0.21 to 0.35 across subreddits

📍 **Content-Driven**: Subreddits focused on roasting, controversy, or explicit content show higher toxicity

📍 **Engagement Correlation**: Some correlation observed between toxicity levels and user engagement metrics

### Practical Applications
- Content moderation strategy development
- Community health monitoring
- Early detection of toxic behavior spread
- Platform policy recommendations

---

## 📦 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
transformers>=4.20.0
torch>=1.10.0
networkx>=2.6.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
python-louvain>=0.15
jupyter>=1.0.0
kagglehub>=0.1.0
tqdm>=4.62.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Out of Memory Error**
- Reduce sample size in the notebook
- Use Google Colab with GPU runtime
- Process data in batches

**2. Model Download Issues**
```python
# Set cache directory
import os
os.environ['TRANSFORMERS_CACHE'] = './models/cache/'
```

**3. Missing Dependencies**
```bash
# Install specific versions
pip install transformers==4.20.0
pip install torch==1.10.0
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Areas for Contribution
- Additional network metrics
- Alternative toxicity detection models
- Enhanced visualizations
- Performance optimization
- Documentation improvements

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@Jashan-Sood](https://github.com/Jashan-Sood)
- LinkedIn: [Your Profile](https://linkedin.com/in/jashan-sood)
- Email: jashansood1711@gmail.com

---

## 🙏 Acknowledgments

- **Dataset**: [1 Million Reddit Comments from 40 Subreddits](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits) by smagnan
- **Model**: [Toxic-BERT](https://huggingface.co/unitary/toxic-bert) by Unitary AI
- **Community Detection**: Louvain algorithm implementation by Thomas Aynaud
- **Inspiration**: Research in online community dynamics and content moderation

---

## 📚 References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
2. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks
3. Kumar, S., et al. (2018). Community Interaction and Conflict on the Web
4. Chandrasekharan, E., et al. (2017). You Can't Stay Here: The Efficacy of Reddit's 2015 Ban Examined Through Hate Speech

---

## 🔗 Useful Links

- [Kaggle Dataset](https://www.kaggle.com/datasets/smagnan/1-million-reddit-comments-from-40-subreddits)
- [Toxic-BERT Model](https://huggingface.co/unitary/toxic-bert)
- [NetworkX Documentation](https://networkx.org/documentation/stable/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

## 📧 Contact & Support

Have questions or suggestions? Feel free to:
- Open an [Issue](https://github.com/Jashan-Sood/reddit-toxicity-detection/issues)
- Start a [Discussion](https://github.com/Jashan-Sood/reddit-toxicity-detection/discussions)
- Email: jashansood1711@gmail.com

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover the project.

---

<div align="center">

**Made with ❤️ for Social Network Analysis Research**

</div>
