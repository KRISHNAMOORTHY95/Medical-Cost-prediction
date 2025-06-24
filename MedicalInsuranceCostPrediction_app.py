# Add these debugging steps before your chart functions:

# 1. Check if DataFrame exists and has data
print("DataFrame shape:", df.shape)
print("DataFrame columns:", df.columns.tolist())
print("First few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())

# 2. Check unique values in categorical columns
if 'region' in df.columns:
    print("Unique region values:", df['region'].unique())
if 'smoker' in df.columns:
    print("Unique smoker values:", df['smoker'].unique())
if 'sex' in df.columns:
    print("Unique sex values:", df['sex'].unique())

# Fixed chart functions with better error handling:

def region_chart(ax):
    # Check if DataFrame and column exist
    if df.empty or 'region' not in df.columns:
        ax.text(0.5, 0.5, 'No region data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Policyholders by Region', fontsize=14, fontweight='bold')
        return
    
    # Handle both numeric and string region values
    region_map = {0: 'Northeast', 1: 'Southeast', 2: 'Southwest', 3: 'Northwest'}
    
    # If regions are already strings, don't map them
    if df['region'].dtype == 'object':
        region_counts = df['region'].value_counts()
    else:
        # Map numeric values to region names
        region_counts = df['region'].map(region_map).value_counts()
    
    if len(region_counts) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Policyholders by Region', fontsize=14, fontweight='bold')
        return
        
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(range(len(region_counts)), region_counts.values, 
                 color=colors[:len(region_counts)], alpha=0.8)
    ax.set_title('Policyholders by Region', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(region_counts)))
    ax.set_xticklabels(region_counts.index, rotation=45)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, region_counts.values)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(region_counts.values) * 0.01,
               f'{int(value)}', ha='center', va='bottom')

def charges_vs_age_chart(ax):
    # Check if required columns exist
    required_cols = ['age', 'charges', 'smoker']
    if df.empty or not all(col in df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'Missing required data (age, charges, smoker)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges vs Age by Smoking Status', fontsize=14, fontweight='bold')
        return
    
    # Remove rows with null values
    clean_df = df[required_cols].dropna()
    
    if clean_df.empty:
        ax.text(0.5, 0.5, 'No valid data after removing nulls', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges vs Age by Smoking Status', fontsize=14, fontweight='bold')
        return
    
    # Separate smokers and non-smokers (handle both 0/1 and True/False)
    non_smokers = clean_df[clean_df['smoker'].isin([0, False, 'no', 'No'])]
    smokers = clean_df[clean_df['smoker'].isin([1, True, 'yes', 'Yes'])]
    
    if len(non_smokers) > 0:
        ax.scatter(non_smokers['age'], non_smokers['charges'], 
                  alpha=0.6, label='Non-Smoker', color='blue', s=30)
    
    if len(smokers) > 0:
        ax.scatter(smokers['age'], smokers['charges'], 
                  alpha=0.6, label='Smoker', color='red', s=30)
    
    ax.set_title('Charges vs Age by Smoking Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Charges ($)')
    
    if len(non_smokers) > 0 or len(smokers) > 0:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No valid smoker data found', ha='center', va='center', transform=ax.transAxes)
        
    ax.grid(True, alpha=0.3)

def smoker_charges_boxplot(ax):
    # Check if required columns exist
    if df.empty or 'smoker' not in df.columns or 'charges' not in df.columns:
        ax.text(0.5, 0.5, 'Missing smoker or charges data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges: Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
        return
    
    # Handle different smoker value formats
    non_smoker_charges = df[df['smoker'].isin([0, False, 'no', 'No'])]['charges'].dropna()
    smoker_charges = df[df['smoker'].isin([1, True, 'yes', 'Yes'])]['charges'].dropna()
    
    if len(non_smoker_charges) == 0 and len(smoker_charges) == 0:
        ax.text(0.5, 0.5, 'No valid charges data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges: Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
        return
        
    # Prepare data for boxplot
    smoker_data = []
    labels = []
    
    if len(non_smoker_charges) > 0:
        smoker_data.append(non_smoker_charges)
        labels.append('Non-Smoker')
        
    if len(smoker_charges) > 0:
        smoker_data.append(smoker_charges)
        labels.append('Smoker')
    
    if smoker_data:
        box = ax.boxplot(smoker_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'salmon']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
    
    ax.set_title('Charges: Smokers vs Non-Smokers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Charges ($)')
    ax.grid(True, alpha=0.3)

def charges_vs_bmi_chart(ax):
    # Check if required columns exist
    required_cols = ['bmi', 'charges', 'smoker']
    if df.empty or not all(col in df.columns for col in required_cols):
        ax.text(0.5, 0.5, 'Missing required data (bmi, charges, smoker)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges vs BMI by Smoking Status', fontsize=14, fontweight='bold')
        return
    
    # Remove rows with null values
    clean_df = df[required_cols].dropna()
    
    if clean_df.empty:
        ax.text(0.5, 0.5, 'No valid data after removing nulls', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges vs BMI by Smoking Status', fontsize=14, fontweight='bold')
        return
    
    # Handle different smoker value formats
    non_smokers = clean_df[clean_df['smoker'].isin([0, False, 'no', 'No'])]
    smokers = clean_df[clean_df['smoker'].isin([1, True, 'yes', 'Yes'])]
    
    if len(non_smokers) > 0:
        ax.scatter(non_smokers['bmi'], non_smokers['charges'], 
                  alpha=0.6, label='Non-Smoker', color='blue', s=30)
    
    if len(smokers) > 0:
        ax.scatter(smokers['bmi'], smokers['charges'], 
                  alpha=0.6, label='Smoker', color='red', s=30)
    
    ax.set_title('Charges vs BMI by Smoking Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('BMI')
    ax.set_ylabel('Charges ($)')
    
    if len(non_smokers) > 0 or len(smokers) > 0:
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No valid smoker data found', ha='center', va='center', transform=ax.transAxes)
        
    ax.grid(True, alpha=0.3)

def gender_charges_chart(ax):
    # Check if required columns exist
    if df.empty or 'sex' not in df.columns or 'charges' not in df.columns:
        ax.text(0.5, 0.5, 'Missing sex or charges data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges by Gender', fontsize=14, fontweight='bold')
        return
    
    # Handle different gender value formats
    female_charges = df[df['sex'].isin([0, 'female', 'Female', 'F'])]['charges'].dropna()
    male_charges = df[df['sex'].isin([1, 'male', 'Male', 'M'])]['charges'].dropna()
    
    if len(female_charges) == 0 and len(male_charges) == 0:
        ax.text(0.5, 0.5, 'No valid gender/charges data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Charges by Gender', fontsize=14, fontweight='bold')
        return
        
    # Prepare data for boxplot
    gender_data = []
    labels = []
    
    if len(female_charges) > 0:
        gender_data.append(female_charges)
        labels.append('Female')
        
    if len(male_charges) > 0:
        gender_data.append(male_charges)
        labels.append('Male')
    
    if gender_data:
        box = ax.boxplot(gender_data, labels=labels, patch_artist=True)
        
        colors = ['pink', 'lightblue']
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
    
    ax.set_title('Charges by Gender', fontsize=14, fontweight='bold')
    ax.set_ylabel('Charges ($)')
    ax.grid(True, alpha=0.3)

# Additional function to load data if needed
def load_and_check_data(file_path):
    """Load CSV and perform basic checks"""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# If you need to reload your data:
# df = load_and_check_data('your_insurance_data.csv')
