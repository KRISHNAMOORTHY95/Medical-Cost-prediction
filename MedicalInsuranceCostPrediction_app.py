# Visualization functions
    def show_distribution_of_charges():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['charges'], kde=True, bins=30, color='teal', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Medical Insurance Charges', fontsize=16, fontweight='bold')
        ax.set_xlabel('Charges ($)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_age_distribution():
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(df['age'], kde=True, bins=20, color='skyblue', alpha=0.7, ax=ax)
        ax.set_title('Distribution of Age', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)
        return fig

    def show_smoker_non_smoker():
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ['Non-Smoker', 'Smoker']
        counts = [sum(df['smoker'] == 0), sum(df['smoker'] == 1)]
        
        bars = ax.bar(labels, counts, color=['lightblue', 'orange'])
        ax.set_title('Smokers vs Non-Smokers', fontsize=16, fontweight='bold')
        ax.set_ylabel('Count')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 5,
                    str(count), ha='center', fontweight='bold')
        return fig

    def show_charges_smokervsnon():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        sns.boxplot(x='smoker', y='charges', data=df, palette='Set2', ax=ax1)
        ax1.set_title('Medical Charges: Smokers vs Non-Smokers', fontweight='bold')
        ax1.set_xlabel('Smoking Status (0=Non-Smoker, 1=Smoker)')
        ax1.set_ylabel('Medical Charges ($)')
        
        # Histogram
        smoker_charges = df[df['smoker'] == 1]['charges']
        non_smoker_charges = df[df['smoker'] == 0]['charges']
        
        ax2.hist(non_smoker_charges, bins=30, alpha=0.7, label='Non-Smoker', color='lightblue')
        ax2.hist(smoker_charges, bins=30, alpha=0.7, label='Smoker', color='orange')
        ax2.set_title('Distribution of Medical Charges', fontweight='bold')
        ax2.set_xlabel('Medical Charges ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        return fig
       
    def show_avg_bmi():
        fig, ax = plt.subplots(figsize=(12, 6))
        average_bmi = df['bmi'].mean()
        sns.histplot(df['bmi'], kde=True, bins=30, color='purple', alpha=0.7, ax=ax)
        ax.axvline(average_bmi, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean BMI: {average_bmi:.2f}')
        ax.set_title('Distribution of BMI', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def show_no_of_policyholders():
        fig, ax = plt.subplots(figsize=(10, 6))
        region_counts = df['region'].value_counts().sort_index()
        region_labels = ['Northeast', 'Southeast', 'Southwest', 'Northwest']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(region_counts)))
        bars = ax.bar(region_labels, region_counts.values, color=colors, alpha=0.8)
        
        ax.set_title('Number of Policyholders by Region', fontsize=16, fontweight='bold')
        ax.set_xlabel('Region', fontsize=12)
        ax.set_ylabel('Number of Policyholders', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, region_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

    def show_charge_age():
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='age', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. Age (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Age', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_bmi_charge():
        fig, ax = plt.subplots(figsize=(12, 8))
        smoker_labels = df['smoker'].map({0: 'Non-Smoker', 1: 'Smoker'})
        sns.scatterplot(x='bmi', y='charges', hue=smoker_labels, data=df.assign(smoker=smoker_labels), 
                       alpha=0.6, palette='Set1', s=60, ax=ax)
        ax.set_title('Charges vs. BMI (Colored by Smoker Status)', fontsize=16, fontweight='bold')
        ax.set_xlabel('BMI', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.legend(title='Smoking Status')
        ax.grid(True, alpha=0.3)
        return fig

    def show_men_women_charge():
        fig, ax = plt.subplots(figsize=(12, 6))
        df['sex_label'] = df['sex'].map({0: 'Female', 1: 'Male'})
        
        sns.boxplot(x='sex_label', y='charges', data=df, palette='pastel', 
                   order=['Female', 'Male'], ax=ax)
        
        ax.set_title('Medical Charges: Male vs Female', fontsize=16, fontweight='bold')
        ax.set_xlabel('Gender', fontsize=12)
        ax.set_ylabel('Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        female_median = df[df['sex_label'] == 'Female']['charges'].median()
        male_median = df[df['sex_label'] == 'Male']['charges'].median()
        
        stats_text = f'Female Median: ${female_median:,.0f}\nMale Median: ${male_median:,.0f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return fig

    def show_correlation_children_charge():
        fig, ax = plt.subplots(figsize=(10, 6))
        children_avg = df.groupby('children')['charges'].mean().reindex(range(0, 6), fill_value=0)
        colors = plt.cm.viridis(np.linspace(0, 1, len(children_avg)))
        
        bars = ax.bar(children_avg.index, children_avg.values, color=colors, alpha=0.8)
        ax.set_title('Average Charges by Number of Children', fontsize=16, fontweight='bold')
        ax.set_xlabel('Number of Children', fontsize=12)
        ax.set_ylabel('Average Charges ($)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, avg_charge in zip(bars, children_avg.values):
            if avg_charge > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                        f'${avg_charge:,.0f}', ha='center', va='bottom')
        return fig

    def show_numeric_features():
        fig, ax = plt.subplots(figsize=(8, 6))
        numeric_cols = ['age', 'bmi', 'children', 'charges']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                   linewidths=0.5, square=True, cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Correlation Between Numeric Features', fontsize=16, fontweight='bold')
        return fig
