## AI Sentiment Analyzer

**AI Sentiment Analyzer** is a Python-based NLP project that analyzes Amazon mobile product reviews to classify sentiment as **Positive, Neutral, or Negative** using TextBlob. It measures both **polarity** (sentiment intensity from -1 to +1) and **subjectivity** (objective vs subjective tone), then produces statistical insights and visualizations.

### Pipeline
1. Load dataset  
2. Clean data (remove duplicates, NaNs, short reviews)  
3. Filter by product (optional)  
4. Compute sentiment scores  
5. Generate visualizations  
6. Export results to CSV  

### Key Features
- Automated sentiment classification  
- Polarity & subjectivity scoring  
- Product-level filtering  
- Sentiment distribution charts  
- Polarity histograms  
- CSV export of analyzed data  

### Tech Stack
- Python (3.7+)  
- pandas, numpy  
- matplotlib, seaborn  
- TextBlob  

### Outputs
- CSV file with sentiment labels & scores  
- Sentiment distribution plot  
- Polarity histogram  

The repository includes a full analysis script, Jupyter notebook workflow, dataset, and modular functions for loading, cleaning, analysis, visualization, and export, with configurable sentiment thresholds.

### *Note:* The dataset was not uploaded due to size limitations. Instead, a download link has been provided. The errors are occurring because the dataset file is not included in the repository.
