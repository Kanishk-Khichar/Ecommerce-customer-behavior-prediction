<h1 align="center">🛒 E-Commerce Customer Purchase Prediction</h1>

<p align="center">
  <b>Machine Learning Project for Predicting Customer Purchase Behavior</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn">
  <img src="https://img.shields.io/badge/Seaborn-Visualization-4C8CBF">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557C">
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
This project analyzes <b>e-commerce customer behavior</b> and uses machine learning classification algorithms to predict whether a customer is likely to make a purchase.
</p>

<p>
The project includes data exploration, visualization, preprocessing, categorical encoding, feature scaling, model training, and evaluation using multiple classification algorithms.
</p>

<h2>🎯 Objectives</h2>

<ul>
  <li>Analyze customer purchasing behavior.</li>
  <li>Explore demographic and e-commerce-related features.</li>
  <li>Visualize customer and transaction patterns.</li>
  <li>Preprocess categorical and numerical data.</li>
  <li>Build machine learning classification models.</li>
  <li>Compare model performance using Accuracy, Precision, Recall, and F1-Score.</li>
</ul>

<h2>📊 Dataset</h2>

<p>
The dataset used in this project is:
</p>

<pre><code>ecommerce_customer_behavior.csv</code></pre>

<p>
The dataset contains <b>5,000 customer records</b> and <b>19 original columns</b>.
There are no missing values in the dataset.
</p>

<h3>Dataset Features</h3>

<table>
  <tr>
    <th>Feature</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Order_ID</td>
    <td>Unique order identifier</td>
  </tr>
  <tr>
    <td>Customer_ID</td>
    <td>Unique customer identifier</td>
  </tr>
  <tr>
    <td>Date</td>
    <td>Order date</td>
  </tr>
  <tr>
    <td>Age</td>
    <td>Customer age</td>
  </tr>
  <tr>
    <td>Gender</td>
    <td>Customer gender</td>
  </tr>
  <tr>
    <td>City</td>
    <td>Customer city</td>
  </tr>
  <tr>
    <td>Product_Category</td>
    <td>Category of purchased product</td>
  </tr>
  <tr>
    <td>Unit_Price</td>
    <td>Price of one product unit</td>
  </tr>
  <tr>
    <td>Quantity</td>
    <td>Number of products purchased</td>
  </tr>
  <tr>
    <td>Discount_Amount</td>
    <td>Discount applied to the order</td>
  </tr>
  <tr>
    <td>Total_Amount</td>
    <td>Total order amount</td>
  </tr>
  <tr>
    <td>Payment_Method</td>
    <td>Payment method used by the customer</td>
  </tr>
  <tr>
    <td>Device_Type</td>
    <td>Device used by the customer</td>
  </tr>
  <tr>
    <td>Session_Duration_Minutes</td>
    <td>Duration of customer session</td>
  </tr>
  <tr>
    <td>Pages_Viewed</td>
    <td>Number of pages viewed</td>
  </tr>
  <tr>
    <td>Is_Returning_Customer</td>
    <td>Whether the customer is returning</td>
  </tr>
  <tr>
    <td>Delivery_Time_Days</td>
    <td>Delivery duration</td>
  </tr>
  <tr>
    <td>Customer_Rating</td>
    <td>Customer rating</td>
  </tr>
  <tr>
    <td>Purchase</td>
    <td>Target variable indicating purchase behavior</td>
  </tr>
</table>

<h2>🔍 Exploratory Data Analysis</h2>

<p>
Several visualizations were created to understand the dataset and customer behavior.
</p>

<ul>
  <li>Purchase distribution</li>
  <li>Gender distribution</li>
  <li>Payment method distribution</li>
  <li>Age distribution based on purchase behavior</li>
  <li>Customer distribution by city</li>
  <li>Product category distribution</li>
</ul>

<h3>👥 Gender Distribution</h3>

<ul>
  <li>Female: <b>2,492</b></li>
  <li>Male: <b>2,435</b></li>
  <li>Other: <b>73</b></li>
</ul>

<h3>💳 Payment Methods</h3>

<ul>
  <li>Credit Card: <b>2,012</b></li>
  <li>Debit Card: <b>1,265</b></li>
  <li>Digital Wallet: <b>965</b></li>
  <li>Bank Transfer: <b>510</b></li>
  <li>Cash on Delivery: <b>248</b></li>
</ul>

<h3>🏙️ Customer Distribution by City</h3>

<ul>
  <li>Istanbul: <b>1,284</b></li>
  <li>Ankara: <b>735</b></li>
  <li>Izmir: <b>600</b></li>
  <li>Bursa: <b>496</b></li>
  <li>Adana: <b>378</b></li>
  <li>Antalya: <b>374</b></li>
  <li>Gaziantep: <b>349</b></li>
  <li>Konya: <b>317</b></li>
  <li>Kayseri: <b>257</b></li>
  <li>Eskisehir: <b>210</b></li>
</ul>

<h3>🛍️ Product Categories</h3>

<ul>
  <li>Sports: <b>667</b></li>
  <li>Electronics: <b>624</b></li>
  <li>Fashion: <b>622</b></li>
  <li>Beauty: <b>621</b></li>
  <li>Home &amp; Garden: <b>621</b></li>
  <li>Food: <b>619</b></li>
  <li>Books: <b>616</b></li>
  <li>Toys: <b>610</b></li>
</ul>

<h2>🧹 Data Preprocessing</h2>

<p>The following preprocessing steps were performed:</p>

<h3>1. Missing Value Check</h3>

<p>
The dataset was checked for missing values using <code>isnull().sum()</code>.
All 19 original columns contained <b>0 missing values</b>.
</p>

<h3>2. Target Encoding</h3>

<p>
The <code>Purchase</code> target column was converted into numerical values using
<b>LabelEncoder</b>.
</p>

<pre><code>le = LabelEncoder()
df["Purchase"] = le.fit_transform(df["Purchase"])</code></pre>

<h3>3. One-Hot Encoding</h3>

<p>
Categorical features were converted into numerical features using
<b>OneHotEncoder</b>.
</p>

<pre><code>cols = [
    "Gender",
    "City",
    "Product_Category",
    "Payment_Method",
    "Device_Type",
    "Is_Returning_Customer"
]

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = ohe.fit_transform(df[cols])</code></pre>

<h3>4. Feature Selection</h3>

<p>
The identifier and date columns were removed before model training.
</p>

<pre><code>X = df.drop(
    columns=["Order_ID", "Customer_ID", "Date", "Purchase"]
)

y = df["Purchase"]</code></pre>

<h3>5. Train-Test Split</h3>

<p>
The dataset was divided into training and testing sets using an
<b>80:20 split</b>.
</p>

<pre><code>X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)</code></pre>

<h3>6. Feature Scaling</h3>

<p>
StandardScaler was used to standardize the numerical features before training the models.
</p>

<pre><code>scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)</code></pre>

<h2>🤖 Machine Learning Models</h2>

<p>Three classification algorithms were implemented:</p>

<ol>
  <li><b>Logistic Regression</b></li>
  <li><b>K-Nearest Neighbors (KNN)</b></li>
  <li><b>Gaussian Naive Bayes</b></li>
</ol>

<h2>📈 Model Performance</h2>

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1-Score</th>
  </tr>
  <tr>
    <td><b>Logistic Regression</b></td>
    <td>89.50%</td>
    <td>88.21%</td>
    <td>90.23%</td>
    <td>89.21%</td>
  </tr>
  <tr>
    <td><b>KNN (k=7)</b></td>
    <td>70.80%</td>
    <td>69.40%</td>
    <td>70.27%</td>
    <td>69.83%</td>
  </tr>
  <tr>
    <td><b>Gaussian Naive Bayes</b></td>
    <td>85.80%</td>
    <td>85.09%</td>
    <td>85.45%</td>
    <td>85.27%</td>
  </tr>
</table>

<h2>📊 Evaluation Metrics</h2>

<ul>
  <li>
    <b>Accuracy:</b> Measures the overall percentage of correct predictions.
  </li>
  <li>
    <b>Precision:</b> Measures how many predicted positive purchases were actually positive.
  </li>
  <li>
    <b>Recall:</b> Measures how many actual positive purchases were correctly identified.
  </li>
  <li>
    <b>F1-Score:</b> Provides a balance between precision and recall.
  </li>
</ul>

<h2>🛠️ Technologies Used</h2>

<table>
  <tr>
    <th>Technology</th>
    <th>Purpose</th>
  </tr>
  <tr>
    <td>Python</td>
    <td>Programming language</td>
  </tr>
  <tr>
    <td>Pandas</td>
    <td>Data manipulation and analysis</td>
  </tr>
  <tr>
    <td>NumPy</td>
    <td>Numerical operations</td>
  </tr>
  <tr>
    <td>Matplotlib</td>
    <td>Data visualization</td>
  </tr>
  <tr>
    <td>Seaborn</td>
    <td>Statistical visualization</td>
  </tr>
  <tr>
    <td>Scikit-learn</td>
    <td>Machine learning and preprocessing</td>
  </tr>
  <tr>
    <td>Jupyter Notebook</td>
    <td>Development environment</td>
  </tr>
</table>

<h2>📁 Project Structure</h2>

<pre><code>eCommerce/
│
├── eCommerce.ipynb
├── ecommerce_customer_behavior.csv
└── README.md
</code></pre>

<h2>⚙️ Installation</h2>

<p>Clone the repository:</p>

<pre><code>git clone &lt;your-repository-url&gt;
cd eCommerce</code></pre>

<p>Install the required Python libraries:</p>

<pre><code>pip install pandas numpy matplotlib seaborn scikit-learn jupyter</code></pre>

<h2>▶️ How to Run</h2>

<ol>
  <li>Download or clone the repository.</li>
  <li>Make sure <code>ecommerce_customer_behavior.csv</code> is in the same directory as the notebook.</li>
  <li>Install the required Python libraries.</li>
  <li>Open <code>eCommerce.ipynb</code> using Jupyter Notebook or VS Code.</li>
  <li>Run the notebook cells sequentially.</li>
</ol>

<h2>🔄 Project Workflow</h2>

<pre><code>Dataset
   ↓
Data Inspection
   ↓
Missing Value Check
   ↓
Exploratory Data Analysis
   ↓
Categorical Encoding
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
</code></pre>

<h2>📌 Key Findings</h2>

<ul>
  <li>The dataset contains <b>5,000 customer records</b>.</li>
  <li>No missing values were found in the original dataset.</li>
  <li>Customer and transaction characteristics were explored through multiple visualizations.</li>
  <li>Categorical variables were converted into numerical features using encoding techniques.</li>
  <li>Three classification algorithms were tested.</li>
  <li>The implemented Logistic Regression model achieved an accuracy of <b>89.50%</b> on the test set.</li>
</ul>

<h2>🚀 Future Improvements</h2>

<ul>
  <li>Perform hyperparameter tuning for all models.</li>
  <li>Test additional algorithms such as Random Forest, Decision Tree, SVM, and Gradient Boosting.</li>
  <li>Perform feature importance analysis.</li>
  <li>Use cross-validation for more robust model evaluation.</li>
  <li>Create an interactive dashboard for customer behavior analysis.</li>
  <li>Deploy the prediction model as a web application.</li>
</ul>

<h2>👨‍💻 Author</h2>

<p>
<b>Your Name: Kanishk Khichar</b><br>
If you like this project, please give me star.
</p>

<h2>⭐ Project Summary</h2>

<p>
This project demonstrates an end-to-end machine learning workflow for
<b>e-commerce customer purchase prediction</b>, starting from data exploration
and visualization and progressing through preprocessing, model training,
prediction, and performance evaluation.
</p>

<hr>

<p align="center">
  <b>🛒 E-Commerce Customer Purchase Prediction | Machine Learning Project</b>
</p>
