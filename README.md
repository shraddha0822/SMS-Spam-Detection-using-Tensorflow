
---

## How to Run
1. Clone repo.  
2. Install dependencies.  
3. Place `spam.csv` in working directory.  
4. Run notebook/script to train & evaluate models.


## Exploratory Data Analysis

- Visualized the distribution of `ham` and `spam` messages using a count plot:  
  ```python
  sns.countplot(x=df['label'])
  plt.show()
