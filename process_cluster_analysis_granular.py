import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr
import argparse
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_and_combine_pca_files(pca_percent, directory):
    file_path = f'{directory}/task_embeddings_cosine_normal_adjusted_similarity_revised_{pca_percent}.csv' if pca_percent is not None else 'task_embeddings_cosine_normal_adjusted_similarity_revised_openai_lg.csv'
    combined_data = pd.read_csv(file_path)
    print(combined_data.head())
    return combined_data

def normalize_similarity_scores(task_rankings_data, nlp):
    task_rankings_data[f'Max_Similarity_Capability_{nlp}_Percent'] = task_rankings_data[f'Max_Similarity_Capability_{nlp}'] * 100
    return task_rankings_data

def penalize_below_threshold(similarity_scores, threshold=0.43, penalty=0.1):
    penalized_scores = similarity_scores.copy()
    penalized_scores[penalized_scores < threshold] *= penalty
    return penalized_scores

def calculate_statistics(task_rankings_data, nlp, threshold=0.43,penalty=0.1):
    # Apply threshold: set similarity below threshold to 0
   
   # Apply penalty to similarity scores below threshold
    penalized = penalize_below_threshold(task_rankings_data[f'Max_Similarity_Capability_{nlp}'], threshold, penalty)

    # Update the main column to reflect penalized similarity
    #task_rankings_data[f'Max_Similarity_Capability_{nlp}'] = penalized



    #task_rankings_data[f'Max_Similarity_Capability_{nlp}'] = task_rankings_data[f'Max_Similarity_Capability_{nlp}'].apply(
    #    lambda x: x if x >= threshold else 0
    #)
    #task_rankings_data[f'Max_Similarity_Capability_{nlp}_Percent'] = task_rankings_data[f'Max_Similarity_Capability_{nlp}'] * 100

    # Update the main column to reflect penalized similarity
    task_rankings_data[f'Max_Similarity_Capability_{nlp}'] = penalized
    task_rankings_data[f'Max_Similarity_Capability_{nlp}_Percent'] = penalized * 100

    occupation_grouped = task_rankings_data.groupby('O*NET-SOC Code').agg(
        **{
            f'Average_Capability_Similarity_{nlp}': (f'Max_Similarity_Capability_{nlp}', 'mean'),
            f'Variance_Capability_Similarity_{nlp}': (f'Max_Similarity_Capability_{nlp}', 'var'),
            f'Average_Capability_Similarity_{nlp}_Percent': (f'Max_Similarity_Capability_{nlp}_Percent', 'mean'),
            f'Variance_Capability_Similarity_{nlp}_Percent': (f'Max_Similarity_Capability_{nlp}_Percent', 'var')
        }
    ).reset_index()
    return occupation_grouped

def process_and_save(pca_percent, nlp, directory):
    task_rankings_data = load_and_combine_pca_files(pca_percent, directory)
    task_rankings_data = normalize_similarity_scores(task_rankings_data, nlp)
    print(f"Columns after normalization: {task_rankings_data.columns}")
    occupation_grouped = calculate_statistics(task_rankings_data, nlp,threshold=0.5)

    combined_df = occupation_grouped.merge(task_rankings_data, on='O*NET-SOC Code', how='inner')
    combined_df = combined_df[['O*NET-SOC Code', 'Title', 
                               f'Variance_Capability_Similarity_{nlp}', f'Average_Capability_Similarity_{nlp}',
                               f'Variance_Capability_Similarity_{nlp}_Percent', f'Average_Capability_Similarity_{nlp}_Percent'
                               ]].drop_duplicates()

    wage_df = pd.read_excel('DATA/wage_2024.xlsx')
    major_occupations_df = pd.read_excel('DATA/major_occupations.xlsx')

    combined_df['Major_OCC_CODE'] = combined_df['O*NET-SOC Code'].apply(lambda x: x.split('-')[0] + '-0000')

    # Join with the major occupations data
    final_df = combined_df.merge(major_occupations_df, left_on='Major_OCC_CODE', right_on='OCC_CODE', how='left', suffixes=('_maj', ''))
    final_df.drop('OCC_CODE', axis=1, inplace=True)

    # Standardize 'O*NET-SOC Code' to match 'OCC_CODE' format
    final_df['O*NET-SOC Code'] = final_df['O*NET-SOC Code'].str.split('.').str[0]

    # Perform the join with wage data
    result_df = pd.merge(final_df, wage_df, left_on='O*NET-SOC Code', right_on='OCC_CODE', how='inner', suffixes=('_maj', ''))

    # Determine the file suffix based on PCA percent
    file_suffix = 'no_pca' if pca_percent is None else f'{pca_percent}'

    # Save the full data to an Excel file
    result_df.to_excel(f'{directory}/np_full_data_{nlp}_{file_suffix}.xlsx', index=False)

    # Calculate similarity scores using fuzzy matching and filter by threshold
    similarity_threshold = 80
    result_df['similarity'] = result_df.apply(lambda x: fuzz.ratio(x['Title'], x['OCC_TITLE']), axis=1)
    df_filtered = result_df[result_df['similarity'] >= similarity_threshold]
    df_filtered.drop('similarity', axis=1, inplace=True)

    # Save the filtered data to an Excel file
    df_filtered.to_excel(f'{directory}/np_full_data_clean_{nlp}_{file_suffix}.xlsx', index=False)

    print(df_filtered.head())

# Function to plot top and bottom 20 occupations by similarity
def plot_top_bottom_20(sorted_data, title, NLP, directory, file_suffix):
    # Select top 20 and bottom 20
    top_20 = sorted_data.head(20).copy()
    top_20['Color'] = 'green'
    bottom_20 = sorted_data.tail(20).copy()
    bottom_20['Color'] = 'red'

    # Combine and create a color mapping
    top_bottom = pd.concat([top_20, bottom_20])
    color_mapping = dict(zip(top_bottom['Title'], top_bottom['Color']))

    # Plot
    plt.figure(figsize=(10, 8), dpi=300)
    sns.barplot(
        x=f'Average_Capability_Similarity_{NLP}_Percent', 
        y='Title', 
        data=top_bottom, 
        palette=color_mapping,
        dodge=False
    )
    plt.xlabel('AI-OCI',fontsize=15)
    plt.ylabel('Occupation Titles',fontsize=15)
    #plt.title(title)
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Top 20'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Bottom 20')
    ], loc='best')
    plt.tight_layout()
    plt.savefig(f'{directory}/top_bottom_20_plot_{NLP}_{file_suffix}.png', format='png', dpi=300)
    plt.show()

# Function to plot correlations
def plot_correlation(data, x, y, title, xlabel, ylabel, filename):
    # Ensure the columns used for plotting are numeric
    data[x] = pd.to_numeric(data[x], errors='coerce')
    data[y] = pd.to_numeric(data[y], errors='coerce')

    # Drop rows with NaN values in relevant columns
    data = data.dropna(subset=[x, y])

    plt.figure(figsize=(10, 6), dpi=300)
    sns.scatterplot(x=x, y=y, data=data, alpha=0.7, edgecolor=None)
    sns.regplot(x=x, y=y, data=data, scatter=False, color='red')
    corr = data[[x, y]].corr().iloc[0, 1]
    plt.text(0.05, 0.95, f'Pearson Corr: {corr:.2f}', ha='left', va='top', transform=plt.gca().transAxes)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

# Function to process data and plot
def process_and_plot(data, nlp_name, directory, file_suffix):
    # Sort data for plotting
    print('nlp_name: ',nlp_name)
    sorted_data = data.sort_values(by=f'Average_Capability_Similarity_{nlp_name}_Percent', ascending=False)

    # Plot top and bottom 20 occupations
    plot_top_bottom_20(
        sorted_data=sorted_data,
        title=f'Top and Bottom 20 Occupations by Max Similarity Capability Percent ({file_suffix})',
        NLP=nlp_name,
        directory=directory,
        file_suffix=file_suffix
    )

    # Plot correlations with wages and employment
    plot_correlation(
        data=sorted_data,
        x=f'Average_Capability_Similarity_{nlp_name}_Percent',
        y='A_MEAN',
        title=f'{nlp_name} Max Similarity Capability vs. Wages ({file_suffix})',
        xlabel='AI-OCI',
        ylabel='Mean Annual Wages',
        filename=f'{directory}/cor_wages_{nlp_name}_{file_suffix}.png'
    )
    plot_correlation(
        data=sorted_data,
        x=f'Average_Capability_Similarity_{nlp_name}_Percent',
        y='TOT_EMP',
        title=f'{nlp_name} Max Similarity Capability vs. Employment ({file_suffix})',
        xlabel='AI-OCI',
        ylabel='Total Employment',
        filename=f'{directory}/cor_employment_{nlp_name}_{file_suffix}.png'
    )

# Merge with AIOE data and compute correlation
def merge_and_compute_aioe_correlation(data, aioe_file, nlp, directory, file_suffix):
    aioe_df = pd.read_csv(aioe_file)[['OCC_CODE', 'AIOE']]
    merged_data = data.merge(aioe_df, on='OCC_CODE', how='inner')
    aioe_corr = merged_data[[f'Average_Capability_Similarity_{nlp}_Percent', 'AIOE']].corr().iloc[0, 1]

    # Plot correlation with AIOE
    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x=['AIOE'], y=[aioe_corr])
    plt.title(f'Correlation with AIOE ({nlp}, {file_suffix})')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(f'{directory}/aioe_correlation_{nlp}_{file_suffix}.png', format='png', dpi=300)
    plt.show()

    print(f'Correlation between AIOE and Max Similarity Capability Percent ({nlp}): {aioe_corr}')

def plot_major_occupations_ranking(data, nlp_name, directory):
    # Ensure the relevant columns are numeric
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )
    data['TOT_EMP'] = pd.to_numeric(data['TOT_EMP'], errors='coerce')
    data['A_MEAN'] = pd.to_numeric(data['A_MEAN'], errors='coerce')

    # Drop rows with NaN values in the relevant columns
    data = data.dropna(
        subset=[f'Average_Capability_Similarity_{nlp_name}_Percent', 'TOT_EMP', 'A_MEAN']
    )

    # Aggregate the data by major occupation and compute the average similarity
    aggregated_data = data.groupby('OCC_TITLE_maj').agg(
        Average_Similarity=(f'Average_Capability_Similarity_{nlp_name}_Percent', 'mean'),
        Std_Similarity=(f'Average_Capability_Similarity_{nlp_name}_Percent', 'std'),
        Average_TOT_EMP=('TOT_EMP', 'mean'),
        Average_A_MEAN=('A_MEAN', 'mean')
    ).reset_index()
    
    # Sort the aggregated data by the average similarity
    sorted_data = aggregated_data.sort_values(by='Average_Similarity', ascending=False)

    # Plot the ranking of major occupations by average similarity with error bars
    plt.figure(figsize=(12, 10))
    ax = sns.barplot(
        y='OCC_TITLE_maj',
        x='Average_Similarity',
        data=sorted_data,
        capsize=0.2,  # Add caps to the error bars
        color='skyblue'
    )

    # Add error bars manually
    ax.errorbar(
        x=sorted_data['Average_Similarity'],
        y=range(len(sorted_data)),
        xerr=sorted_data['Std_Similarity'],
        fmt='none',
        color='black',
        capsize=3
    )

    plt.xlabel('AI-OCI')
    plt.ylabel('Major Occupation Title')
    plt.tight_layout()
    plt.savefig(f'{directory}/major_occupations_ranking_{nlp_name}.png', format='png', dpi=300)
    plt.show()

def plot_correlation_by_era(data, nlp_name, directory):
    # Ensure necessary columns are numeric
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )
    data['TOT_EMP'] = pd.to_numeric(data['TOT_EMP'], errors='coerce')
    data['A_MEAN'] = pd.to_numeric(data['A_MEAN'], errors='coerce')
    def get_axis_limits(data, metric):
        return data[metric].min(), data[metric].max()

    # Calculate global limits
    min_wage, max_wage = get_axis_limits(data, 'A_MEAN')
    
    # Define eras
    eras = {
        'Pre-Covid Era': {'start': data['year'].min(), 'end': 2019, 'color': 'skyblue'},
        'Covid Era': {'start': 2020, 'end': 2021, 'color': 'sandybrown'},
        'LLM Market Adoption Era': {'start': 2022, 'end': data['year'].max(), 'color': 'lightgreen'}
    }
    print('max year: ',data['year'].max()) 
    # Iterate through each era
    for era_name, era_details in eras.items():
        era_data = data[(data['year'] >= era_details['start']) & (data['year'] <= era_details['end'])]

        # Bin data into 20 equal-sized bins
        era_data['AIOCI_Bin'] = pd.qcut(
            era_data[f'Average_Capability_Similarity_{nlp_name}_Percent'], q=20, labels=False
        )

        # Group by bins and calculate means
        binned_data = era_data.groupby('AIOCI_Bin').agg(
            Mean_AIOCI=(f'Average_Capability_Similarity_{nlp_name}_Percent', 'mean'),
            Mean_Wages=('A_MEAN', 'mean'),
            Mean_Employment=('TOT_EMP', 'mean')
        ).reset_index()
        
        # Calculate global limits
        #min_wage, max_wage = get_axis_limits(binned_data, 'Mean_Wages')

        # Plot wages vs. AIOCI
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='Mean_AIOCI', y='Mean_Wages', data=binned_data, color=era_details['color'], s=50
        )
        sns.regplot(
            x='Mean_AIOCI', y='Mean_Wages', data=binned_data, scatter=False, color='red'
        )

        # Calculate and add correlation text
        correlation_wages = binned_data[['Mean_AIOCI', 'Mean_Wages']].corr().iloc[0, 1]
        plt.text(
            x=binned_data['Mean_AIOCI'].min(),
            y=binned_data['Mean_Wages'].max() * 0.95,
            s=f'Corr = {correlation_wages:.2f}',
            color='black',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

        #plt.title(f'Wages vs. AIOCI ({era_name})')
        plt.ylim(30000,140000)
        plt.xlabel('AI-OCI')
        plt.ylabel('Mean Wages (USD)')
        plt.tight_layout()
        plt.savefig(f'{directory}/wages_vs_aioci_{era_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()

        print(f"{era_name}: Correlation = {correlation_wages:.2f}")        

        # Plot employment vs. AIOCI
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x='Mean_AIOCI', y='Mean_Employment', data=binned_data, color=era_details['color'], s=50
        )
        sns.regplot(
            x='Mean_AIOCI', y='Mean_Employment', data=binned_data, scatter=False, color='red'
        )
        # Calculate and add correlation text
        correlation_employment = binned_data[['Mean_AIOCI', 'Mean_Employment']].corr().iloc[0, 1]
        plt.text(
            x=binned_data['Mean_AIOCI'].min(),
            y=binned_data['Mean_Employment'].max() * 0.95,
            s=f'Corr = {correlation_employment:.2f}',
            color='black',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )
        #plt.title(f'Employment vs. AIOCI ({era_name})')
        plt.xlabel('AI-OCI')
        plt.ylabel('Mean Employment')
        plt.tight_layout()
        plt.savefig(f'{directory}/employment_vs_aioci_{era_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.show()

        # Debug placement information
        print(f"{era_name}: Correlation = {correlation_employment:.2f}")
        #print(f"Text Position: x={text_x}, y={text_y}")


def plot_overall_correlation(data, x_col, y_col, x_label, y_label, title, filename,directory):
    """
    Plots the overall correlation between x_col and y_col, with the correlation value added to the plot.
    Handles non-numeric values by coercing them to NaN and dropping them.
    """
    # Ensure columns are numeric, and handle invalid values
    data[x_col] = pd.to_numeric(data[x_col], errors='coerce')
    data[y_col] = pd.to_numeric(data[y_col], errors='coerce')
    
    # Drop rows with NaN values in the columns used for correlation
    cleaned_data = data.dropna(subset=[x_col, y_col])

    if cleaned_data.empty:
        print(f"Insufficient data for plotting {title}.")
        return

    # Calculate Pearson correlation
    correlation = cleaned_data[[x_col, y_col]].corr(method='pearson').iloc[0, 1]

    # Plot scatter plot with regression line
    plt.figure(figsize=(10, 6), dpi=300)
    sns.scatterplot(x=x_col, y=y_col, data=cleaned_data, alpha=0.7)
    sns.regplot(x=x_col, y=y_col, data=cleaned_data, scatter=False, color='red')

    # Add correlation value to the plot
    plt.text(
        0.05,
        0.95,
        f'Correlation: {correlation:.2f}',
        ha='left',
        va='top',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'{directory}/{filename}', format='png', dpi=300)
    #plt.savefig(filename, format='png', dpi=300)
    plt.show()

def plot_combined_correlation_by_era(data, nlp_name, directory):
    """
    Create combined plots showing correlations for all eras in one plot:
    - One for Wages vs. AI-OCI
    - One for Employment vs. AI-OCI
    """
    # Ensure necessary columns are numeric
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )
    data['TOT_EMP'] = pd.to_numeric(data['TOT_EMP'], errors='coerce')
    data['A_MEAN'] = pd.to_numeric(data['A_MEAN'], errors='coerce')

    # Define eras
    eras = {
        'Pre-Covid Era': {'start': data['year'].min(), 'end': 2019, 'color': 'skyblue'},
        'Covid Era': {'start': 2020, 'end': 2021, 'color': 'sandybrown'},
        'LLM Market Adoption Era': {'start': 2022, 'end': data['year'].max(), 'color': 'lightgreen'}
    }
    print('test2 max year :',data['year'].max())
    # Create combined plot for wages
    plt.figure(figsize=(10, 7))
    for era_name, era_details in eras.items():
        era_data = data[(data['year'] >= era_details['start']) & (data['year'] <= era_details['end'])]

        # Bin data into 20 equal-sized bins
        era_data['AIOCI_Bin'] = pd.qcut(
            era_data[f'Average_Capability_Similarity_{nlp_name}_Percent'], q=20, labels=False
        )

        # Group by bins and calculate means
        binned_data = era_data.groupby('AIOCI_Bin').agg(
            Mean_AIOCI=(f'Average_Capability_Similarity_{nlp_name}_Percent', 'mean'),
            Mean_Wages=('A_MEAN', 'mean')
        ).reset_index()

        # Scatter plot and regression line
        sns.scatterplot(
            x='Mean_AIOCI', y='Mean_Wages', data=binned_data, color=era_details['color'], label=f'{era_name}', s=50
        )
        sns.regplot(
            x='Mean_AIOCI', y='Mean_Wages', data=binned_data, scatter=False, color=era_details['color'], ci=None
        )

        # Calculate and add correlation text
        correlation_wages = binned_data[['Mean_AIOCI', 'Mean_Wages']].corr().iloc[0, 1]
        plt.text(
            x=binned_data['Mean_AIOCI'].min(),
            y=binned_data['Mean_Wages'].max() * 0.95,
            s=f'{era_name}: Corr = {correlation_wages:.2f}',
            color=era_details['color'],
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    plt.title('Wages vs. AI-OCI Across Eras')
    plt.xlabel('AI-OCI')
    plt.ylabel('Mean Wages (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/combined_wages_vs_aioci.png', dpi=300)
    plt.show()

    # Create combined plot for employment
    plt.figure(figsize=(10, 7))
    for era_name, era_details in eras.items():
        era_data = data[(data['year'] >= era_details['start']) & (data['year'] <= era_details['end'])]

        # Bin data into 20 equal-sized bins
        era_data['AIOCI_Bin'] = pd.qcut(
            era_data[f'Average_Capability_Similarity_{nlp_name}_Percent'], q=20, labels=False
        )

        # Group by bins and calculate means
        binned_data = era_data.groupby('AIOCI_Bin').agg(
            Mean_AIOCI=(f'Average_Capability_Similarity_{nlp_name}_Percent', 'mean'),
            Mean_Employment=('TOT_EMP', 'mean')
        ).reset_index()

        # Scatter plot and regression line
        sns.scatterplot(
            x='Mean_AIOCI', y='Mean_Employment', data=binned_data, color=era_details['color'], label=f'{era_name}', s=50
        )
        sns.regplot(
            x='Mean_AIOCI', y='Mean_Employment', data=binned_data, scatter=False, color=era_details['color'], ci=None
        )

        # Calculate and add correlation text
        correlation_employment = binned_data[['Mean_AIOCI', 'Mean_Employment']].corr().iloc[0, 1]
        plt.text(
            x=binned_data['Mean_AIOCI'].min(),
            y=binned_data['Mean_Employment'].max() * 0.95,
            s=f'{era_name}: Corr = {correlation_employment:.2f}',
            color=era_details['color'],
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

    plt.title('Employment vs. AI-OCI Across Eras')
    plt.xlabel('AI-OCI')
    plt.ylabel('Mean Employment')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{directory}/combined_employment_vs_aioci.png', dpi=300)
    plt.show()

def plot_relative_change_grouped_bar(overtime_df, directory):
    """
    Plot a grouped bar chart for relative changes in employment across eras for major occupations.

    Parameters:
    - overtime_df: DataFrame containing employment data with 'group' == 'major'.
    - directory: Path to save the plot.
    """
    # Filter for 'major' group
    major_data = overtime_df[overtime_df['group'] == 'major']
    major_data['tot_emp_per_year'] = pd.to_numeric(major_data['tot_emp_per_year'], errors='coerce')
    
    major_data['OCC_TITLE'] = (
        major_data['OCC_TITLE']
    .str.title()  # Title case
    .str.replace('Occupations', '', regex=False)  # Remove "Occupations"
    .str.strip()  # Remove leading/trailing whitespace
    .str.replace(r'\b(Services|Systems|Technicians)\b', 'Service', regex=True)  # Handle plurals
    )   
    

    # Create a mapping of `OCC_CODE` to the first valid `OCC_TITLE`
    title_mapping = (
        major_data[['OCC_CODE', 'OCC_TITLE']]
        .drop_duplicates(subset=['OCC_CODE'])
        .set_index('OCC_CODE')['OCC_TITLE']
    )
    major_data['OCC_TITLE'] = major_data['OCC_TITLE'].str.title()  # Converts to title case
    major_data['OCC_TITLE'] = major_data['OCC_TITLE'].str.title().str.replace('Occupations', '').str.strip()
    major_data = major_data.drop_duplicates(subset=['OCC_TITLE', 'year', 'tot_emp_per_year'])
    # Define the eras
    pre_covid_era = major_data[major_data['year'] <= 2019]
    covid_era = major_data[(major_data['year'] >= 2020) & (major_data['year'] <= 2021)]
    llm_era = major_data[major_data['year'] >= 2022]

    # Aggregate the data for each era by OCC_TITLE
    pre_covid_avg = pre_covid_era.groupby('OCC_CODE')['tot_emp_per_year'].mean().rename('TOT_EMP_Pre_COVID')
    covid_avg = covid_era.groupby('OCC_CODE')['tot_emp_per_year'].mean().rename('TOT_EMP_COVID')
    llm_avg = llm_era.groupby('OCC_CODE')['tot_emp_per_year'].mean().rename('TOT_EMP_LLM')

    # Merge the aggregated data into a single DataFrame
    era_comparison = pd.concat([pre_covid_avg, covid_avg, llm_avg], axis=1).reset_index()

    # Merge the aggregated data into a single DataFrame
    era_comparison = pd.concat([pre_covid_avg, covid_avg, llm_avg], axis=1).reset_index()

    # Map titles for display using `OCC_CODE`
    era_comparison['OCC_TITLE'] = era_comparison['OCC_CODE'].map(title_mapping)

    # Exclude "All" from the data
    era_comparison = era_comparison[era_comparison['OCC_TITLE'] != "All"]


    # Calculate relative changes
    era_comparison['Relative_Change_Pre_to_COVID'] = (
        (era_comparison['TOT_EMP_COVID'] - era_comparison['TOT_EMP_Pre_COVID']) /
        era_comparison['TOT_EMP_Pre_COVID'] * 100
    )
    era_comparison['Relative_Change_COVID_to_LLM'] = (
        (era_comparison['TOT_EMP_LLM'] - era_comparison['TOT_EMP_COVID']) /
        era_comparison['TOT_EMP_COVID'] * 100
    )

    # Fill missing values (e.g., NaN) with zeros for plotting
    era_comparison = era_comparison.fillna(0)

    # Melt the DataFrame for plotting
    plot_data = era_comparison.melt(
        id_vars='OCC_TITLE',
        value_vars=['Relative_Change_Pre_to_COVID', 'Relative_Change_COVID_to_LLM'],
        var_name='Era_Comparison',
        value_name='Relative_Change'
    )

    # Plot the grouped bar chart
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=plot_data,
        x='OCC_TITLE',
        y='Relative_Change',
        hue='Era_Comparison',
        palette='viridis'
    )
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add a horizontal reference line at 0
    plt.xlabel('Major Occupation Title', fontsize=12)
    plt.ylabel('Relative Change (%)', fontsize=12)
    plt.title('Relative Changes in Employment Across Eras (Major Occupations)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='Era Comparison', loc='upper left')
    plt.tight_layout()

    # Save the plot
    filename = f'{directory}/relative_change_grouped_bar_major.png'
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

    print(f"Grouped bar chart saved at {filename}")

def plot_relative_change_grouped_bar_with_error(overtime_df, directory):
    """
    Plot a grouped bar chart for relative changes in employment across eras for major occupations,
    with error bars indicating variability (e.g., standard deviation).

    Parameters:
    - overtime_df: DataFrame containing employment data with 'group' == 'major'.
    - directory: Path to save the plot.
    """
    # Preprocess data (same as earlier)
    major_data = overtime_df[overtime_df['group'] == 'major']
    major_data = major_data.drop_duplicates(subset=['OCC_CODE', 'year', 'tot_emp_per_year'])
    major_data['OCC_TITLE'] = (
        major_data['OCC_TITLE']
        .str.title()
        .str.replace('Occupations', '', regex=False)
        .str.strip()
    )

    title_mapping = (
        major_data[['OCC_CODE', 'OCC_TITLE']]
        .drop_duplicates(subset=['OCC_CODE'])
        .set_index('OCC_CODE')['OCC_TITLE']
    )
    major_data['tot_emp_per_year'] = pd.to_numeric(major_data['tot_emp_per_year'], errors='coerce')

    pre_covid_era = major_data[major_data['year'] <= 2019]
    covid_era = major_data[(major_data['year'] >= 2020) & (major_data['year'] <= 2021)]
    llm_era = major_data[major_data['year'] >= 2022]

    pre_covid_stats = pre_covid_era.groupby('OCC_CODE')['tot_emp_per_year'].agg(['mean', 'std']).rename(
        columns={'mean': 'TOT_EMP_Pre_COVID', 'std': 'STD_Pre_COVID'}
    )
    covid_stats = covid_era.groupby('OCC_CODE')['tot_emp_per_year'].agg(['mean', 'std']).rename(
        columns={'mean': 'TOT_EMP_COVID', 'std': 'STD_COVID'}
    )
    llm_stats = llm_era.groupby('OCC_CODE')['tot_emp_per_year'].agg(['mean', 'std']).rename(
        columns={'mean': 'TOT_EMP_LLM', 'std': 'STD_LLM'}
    )

    era_comparison = pd.concat([pre_covid_stats, covid_stats, llm_stats], axis=1).reset_index()
    era_comparison['OCC_TITLE'] = era_comparison['OCC_CODE'].map(title_mapping)
    era_comparison = era_comparison[era_comparison['OCC_TITLE'] != "All"]

    era_comparison['Relative_Change_Pre_to_COVID'] = (
        (era_comparison['TOT_EMP_COVID'] - era_comparison['TOT_EMP_Pre_COVID']) /
        era_comparison['TOT_EMP_Pre_COVID'] * 100
    )
    era_comparison['Relative_Change_COVID_to_LLM'] = (
        (era_comparison['TOT_EMP_LLM'] - era_comparison['TOT_EMP_COVID']) /
        era_comparison['TOT_EMP_COVID'] * 100
    )

    era_comparison['Error_Pre_to_COVID'] = era_comparison['STD_Pre_COVID'] / era_comparison['TOT_EMP_Pre_COVID'] * 100
    era_comparison['Error_COVID_to_LLM'] = era_comparison['STD_COVID'] / era_comparison['TOT_EMP_COVID'] * 100

    era_comparison = era_comparison.fillna(0)
    print(era_comparison.head())
    print(era_comparison.columns)
    print(era_comparison[['Relative_Change_Pre_to_COVID', 'Relative_Change_COVID_to_LLM']].head())

    plot_data = era_comparison.melt(
        id_vars=['OCC_TITLE'],
        value_vars=['Relative_Change_Pre_to_COVID', 'Relative_Change_COVID_to_LLM'],
        var_name='Era_Comparison',
        value_name='Relative_Change'
    )
    print('plot_data',plot_data) 

    error_data = era_comparison.melt(
        id_vars=['OCC_TITLE'],
        value_vars=['Error_Pre_to_COVID', 'Error_COVID_to_LLM'],
        var_name='Era_Comparison',
        value_name='Error'
    )
    print('error data:',error_data)
    print("Value Variables Passed to Melt:", ['Relative_Change_Pre_to_COVID', 'Relative_Change_COVID_to_LLM'])
    print(era_comparison[['Relative_Change_Pre_to_COVID', 'Relative_Change_COVID_to_LLM']].isna().sum())

    # Define mapping to align Era_Comparison in error_data with plot_data
    error_mapping = {
        "Error_Pre_to_COVID": "Relative_Change_Pre_to_COVID",
        "Error_COVID_to_LLM": "Relative_Change_COVID_to_LLM",
    }

    # Apply the mapping to the Era_Comparison column in error_data
    error_data["Era_Comparison"] = error_data["Era_Comparison"].replace(error_mapping)

    # Now merge with standardized Era_Comparison
    #plot_data = plot_data.merge(error_data, on=["OCC_TITLE", "Era_Comparison"])

    plot_data = plot_data.merge(error_data, on=['OCC_TITLE', 'Era_Comparison'])
    print('plot_after_merge:',plot_data)
    print(plot_data.head())
    print(plot_data[['Relative_Change', 'Error']].dtypes)
    print(plot_data[['Relative_Change', 'Error']].describe())
    print(plot_data['Era_Comparison'].unique())


    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=plot_data,
        x='OCC_TITLE',
        y='Relative_Change',
        hue='Era_Comparison',
        palette='viridis',
        ci=None
    )
    #for index, row in plot_data.iterrows():
     #   plt.errorbar(
      #      x=index,
       #     y=row['Relative_Change'],
        #    yerr=row['Error'],
         #   fmt='none',
          #  capsize=3,
           # color='black'
       # )
    from matplotlib.ticker import MaxNLocator

    # Add error bars
    bar_positions = []
    for p in plt.gca().patches:  # Collect the positions of the bars
        bar_positions.append(p.get_x() + p.get_width() / 2)

    for i, (pos, row) in enumerate(zip(bar_positions, plot_data.iterrows())):
        plt.errorbar(
        x=pos,
        y=row[1]['Relative_Change'],
        yerr=row[1]['Error'],
        fmt='none',
        capsize=3,
        color='black'
        )   
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Major Occupation Title', fontsize=12)
    plt.ylabel('Relative Change (%)', fontsize=12)
    # plt.title('Relative Changes in Employment Across Eras (Major Occupations)', fontsize=14)
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.legend(title='Era Comparison', loc='upper left')
    plt.tight_layout()

    filename = f'{directory}/relative_change_grouped_bar_with_error_major.png'
    plt.savefig(filename, format='png', dpi=300)
    plt.show()

    print(f"Grouped bar chart with error bars saved at {filename}")

def plot_top_aioci_occupation_trends_remove(data, nlp_name, directory, top_n=10):
    """
    Plot employment and wage trends across eras for top N occupations with highest AI-OCI scores.
    """
    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['A_MEAN'] = pd.to_numeric(data['A_MEAN'], errors='coerce')
    data['TOT_EMP'] = pd.to_numeric(data['TOT_EMP'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Identify top N occupations by AI-OCI
    top_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates()
        .nlargest(top_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in top_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        occ_data = data[data['OCC_CODE'] == occ_code]

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=occ_data, x='year', y='A_MEAN', marker='o')
        plt.title(f'Wage Trends for {occ_title}')
        plt.ylabel('Mean Wage (USD)')
        plt.xlabel('Year')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{directory}/wage_trend_{occ_code}.png', dpi=300)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=occ_data, x='year', y='TOT_EMP', marker='o', color='orange')
        plt.title(f'Employment Trends for {occ_title}')
        plt.ylabel('Total Employment')
        plt.xlabel('Year')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{directory}/employment_trend_{occ_code}.png', dpi=300)
        plt.close()


def plot_top_aioci_occupation_trends(data, nlp_name, directory, top_n=10):
    """
    Plot side-by-side employment and wage trends for top N occupations with highest AI-OCI scores.
    """
    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Drop invalid data
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    # Identify top N occupations by AI-OCI
    top_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates(subset=['OCC_CODE'])
        .nlargest(top_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in top_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        safe_title = occ_title.replace(' ', '_').replace('/', '_')
        occ_data = data[data['OCC_CODE'] == occ_code].sort_values('year')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

        # --- Wage Trend ---
        axes[0].plot(
            occ_data['year'].to_numpy(), 
            occ_data['a_mean_per_year'].to_numpy(), 
            marker='o', 
            color='steelblue'
        )
        axes[0].set_title('Wage Trends')
        axes[0].set_ylabel('Mean Wage (USD)')
        axes[0].set_xlabel('Year')
        axes[0].axvline(2020, linestyle='--', color='gray', label='COVID Start')
        axes[0].axvline(2022, linestyle='--', color='green', label='LLM Adoption')
        axes[0].legend()

        # --- Employment Trend ---
        axes[1].plot(
            occ_data['year'].to_numpy(), 
            occ_data['tot_emp_per_year'].to_numpy(), 
            marker='o', 
            color='darkorange'
        )
        axes[1].set_title('Employment Trends')
        axes[1].set_ylabel('Total Employment')
        axes[1].set_xlabel('Year')
        axes[1].axvline(2020, linestyle='--', color='gray', label='COVID Start')
        axes[1].axvline(2022, linestyle='--', color='green', label='LLM Adoption')
        axes[1].legend()

        fig.suptitle(f'{occ_title} (AI-OCI: {row[f"Average_Capability_Similarity_{nlp_name}_Percent"]:.1f}%)', fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join(directory, f'combined_trend_{safe_title}_{occ_code}.png')
        plt.savefig(output_path)
        plt.close()
        print(f"âœ… Saved trend plot for {occ_title} â†’ {output_path}")

def plot_percentage_change_for_top_occupations(data, nlp_name, top_n=5, directory=".", start_year=2012):
    """
    Plots the percentage change in wages and employment for top N occupations based on AI-OCI scores.

    Parameters:
        data (DataFrame): Input DataFrame with yearly wage and employment information.
        nlp_name (str): Name of the NLP model (e.g., 'openai').
        top_n (int): Number of top occupations to plot.
        directory (str): Path to save the figures.
        start_year (int): Year from which to begin analysis.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os


    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Drop missing values
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    # Identify top N occupations by AI-OCI
    top_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates(subset=['OCC_CODE'])
        .nlargest(top_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in top_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        occ_data = data[data['OCC_CODE'] == occ_code].sort_values('year')

        # Check for base year
        if start_year not in occ_data['year'].values:
            print(f"âš ï¸ Skipping {occ_title} ({occ_code}) â€” no data for base year {start_year}.")
            continue

        # Compute percent changes
        base_wage = occ_data.loc[occ_data['year'] == start_year, 'a_mean_per_year'].values[0]
        base_emp = occ_data.loc[occ_data['year'] == start_year, 'tot_emp_per_year'].values[0]
        occ_data['Wage_Pct_Change'] = (occ_data['a_mean_per_year'] - base_wage) / base_wage * 100
        occ_data['Employment_Pct_Change'] = (occ_data['tot_emp_per_year'] - base_emp) / base_emp * 100

        # Drop rows with NaNs after computation
        occ_data = occ_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        #fig.suptitle(f'{occ_title} (AI-OCI: {row[f"Average_Capability_Similarity_{nlp_name}_Percent"]:.1f}%)', fontsize=14)

        # Wage %
        axes[0].plot(occ_data['year'].to_numpy(), occ_data['Wage_Pct_Change'].to_numpy(), marker='o', color='steelblue')
        axes[0].set_title("Wage Change (%)")
        axes[0].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[0].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[0].set_ylabel("Percent Change")
        axes[0].legend()
        axes[0].grid(True)

        # Employment %
        axes[1].plot(occ_data['year'].to_numpy(), occ_data['Employment_Pct_Change'].to_numpy(), marker='o', color='orange')
        axes[1].set_title("Employment Change (%)")
        axes[1].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[1].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[1].set_ylabel("Percent Change")
        axes[1].legend()
        axes[1].grid(True)

        for ax in axes:
            ax.set_xlabel("Year")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_title = occ_title.replace(' ', '_').replace('/', '_')
        fig_path = f"{directory}/pct_change_trends_{safe_title}_{occ_code}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"âœ… Saved: {fig_path}")


def plot_percentage_change_for_bottom_occupations(data, nlp_name, bottom_n=5, directory=".", start_year=2012):
    """
    Plots the percentage change in wages and employment for bottom N occupations based on AI-OCI scores.

    Parameters:
        data (DataFrame): Input DataFrame with yearly wage and employment information.
        nlp_name (str): Name of the NLP model (e.g., 'openai').
        bottom_n (int): Number of bottom occupations to plot.
        directory (str): Path to save the figures.
        start_year (int): Year from which to begin analysis.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Drop missing values
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    # Identify bottom N occupations by AI-OCI
    bottom_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates(subset=['OCC_CODE'])
        .nsmallest(bottom_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in bottom_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        occ_data = data[data['OCC_CODE'] == occ_code].sort_values('year')

        # Check for base year
        if start_year not in occ_data['year'].values:
            print(f"âš ï¸ Skipping {occ_title} ({occ_code}) â€” no data for base year {start_year}.")
            continue

        # Compute percent changes
        base_wage = occ_data.loc[occ_data['year'] == start_year, 'a_mean_per_year'].values[0]
        base_emp = occ_data.loc[occ_data['year'] == start_year, 'tot_emp_per_year'].values[0]
        occ_data['Wage_Pct_Change'] = (occ_data['a_mean_per_year'] - base_wage) / base_wage * 100
        occ_data['Employment_Pct_Change'] = (occ_data['tot_emp_per_year'] - base_emp) / base_emp * 100

        # Drop rows with NaNs after computation
        occ_data = occ_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        #fig.suptitle(f'{occ_title} (AI-OCI: {row[f"Average_Capability_Similarity_{nlp_name}_Percent"]:.1f}%)', fontsize=14)

        # Wage %
        axes[0].plot(occ_data['year'].to_numpy(), occ_data['Wage_Pct_Change'].to_numpy(), marker='o', color='steelblue')
        axes[0].set_title("Wage Change (%)")
        axes[0].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[0].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[0].set_ylabel("Percent Change")
        axes[0].legend()
        axes[0].grid(True)

        # Employment %
        axes[1].plot(occ_data['year'].to_numpy(), occ_data['Employment_Pct_Change'].to_numpy(), marker='o', color='orange')
        axes[1].set_title("Employment Change (%)")
        axes[1].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[1].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[1].set_ylabel("Percent Change")
        axes[1].legend()
        axes[1].grid(True)

        for ax in axes:
            ax.set_xlabel("Year")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_title = occ_title.replace(' ', '_').replace('/', '_')
        fig_path = f"{directory}/pct_change_trends_bottom_{safe_title}_{occ_code}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"âœ… Saved: {fig_path}")

def plot_yoy_change_for_bottom_occupations(data, nlp_name, bottom_n=5, directory=".", title_prefix=""):
    """
    Plots year-over-year percentage change in wages and employment for bottom-N occupations by AI-OCI.

    Parameters:
        data (DataFrame): Data with 'year', 'a_mean_per_year', 'tot_emp_per_year', and AI-OCI.
        nlp_name (str): Column suffix used for AI-OCI.
        bottom_n (int): Number of bottom-ranked occupations by AI-OCI to include.
        directory (str): Folder path to save output plots.
        title_prefix (str): Optional prefix for plot titles.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure numeric types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Drop missing values
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    # Get bottom N occupations by AI-OCI
    bottom_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates(subset=['OCC_CODE'])
        .nsmallest(bottom_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in bottom_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        occ_data = data[data['OCC_CODE'] == occ_code].sort_values('year')

        # Compute YoY % change
        occ_data['Wage_Pct_Change'] = occ_data['a_mean_per_year'].pct_change() * 100
        occ_data['Employment_Pct_Change'] = occ_data['tot_emp_per_year'].pct_change() * 100
        occ_data = occ_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

        # Skip if not enough points
        if occ_data.shape[0] < 2:
            continue

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
        fig.suptitle(f'{title_prefix}{occ_title} (AI-OCI: {row[f"Average_Capability_Similarity_{nlp_name}_Percent"]:.1f}%)', fontsize=14)

        # Wage YoY %
        axes[0].set_ylim(-15, 15)
        #sns.lineplot(ax=axes[0], data=occ_data, x='year', y='Wage_Pct_Change', marker='o', color='steelblue')
        axes[0].plot(occ_data['year'].to_numpy(), occ_data['Wage_Pct_Change'].to_numpy(), marker='o', color='steelblue')
        axes[0].set_title("Wage YoY Change (%)")
        axes[0].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[0].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[0].set_ylabel("Percent Change")
        axes[0].legend()
        axes[0].grid(True)

        # Employment YoY %
        axes[1].set_ylim(-15, 15)
        #sns.lineplot(ax=axes[1], data=occ_data, x='year', y='Employment_Pct_Change', marker='o', color='darkorange')
        axes[1].plot(occ_data['year'].to_numpy(), occ_data['Employment_Pct_Change'].to_numpy(), marker='o', color='darkorange')

        axes[1].set_title("Employment YoY Change (%)")
        axes[1].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[1].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[1].set_ylabel("Percent Change")
        axes[1].legend()
        axes[1].grid(True)

        for ax in axes:
            ax.set_xlabel("Year")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_title = occ_title.replace(' ', '_').replace('/', '_')
        fig_path = os.path.join(directory, f'yoy_pct_change_{safe_title}_{occ_code}.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"âœ… Saved: {fig_path}")


def plot_yoy_change_for_top_occupations(data, nlp_name, top_n=5, directory=".", title_prefix="Top: "):
    """
    Plots year-over-year percentage change in wages and employment for top N occupations based on AI-OCI scores.

    Parameters:
        data (DataFrame): Input DataFrame with yearly wage and employment information.
        nlp_name (str): Name of the NLP model (e.g., 'openai').
        top_n (int): Number of top occupations to plot.
        directory (str): Path to save the figures.
        title_prefix (str): Optional prefix for plot titles.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Drop missing values
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    # Identify top N occupations by AI-OCI
    top_occupations = (
        data[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
        .drop_duplicates(subset=['OCC_CODE'])
        .nlargest(top_n, f'Average_Capability_Similarity_{nlp_name}_Percent')
    )

    for _, row in top_occupations.iterrows():
        occ_code = row['OCC_CODE']
        occ_title = row['Title']
        occ_data = data[data['OCC_CODE'] == occ_code].sort_values('year')

        # Compute YoY percent change
        occ_data['Wage_Pct_Change'] = occ_data['a_mean_per_year'].pct_change() * 100
        occ_data['Employment_Pct_Change'] = occ_data['tot_emp_per_year'].pct_change() * 100
        occ_data = occ_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

        # Skip if not enough points
        if occ_data.shape[0] < 2:
            continue

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True)
        fig.suptitle(f'{title_prefix}{occ_title} (AI-OCI: {row[f"Average_Capability_Similarity_{nlp_name}_Percent"]:.1f}%)', fontsize=14)

        # Wage %
        #sns.lineplot(ax=axes[0], data=occ_data, x='year', y='Wage_Pct_Change', marker='o', color='steelblue')
        #sns.lineplot(ax=axes[1], data=occ_data, x='year', y='Employment_Pct_Change', marker='o', color='darkorange')
        axes[0].plot(occ_data['year'].to_numpy(), occ_data['Wage_Pct_Change'].to_numpy(), marker='o', color='steelblue')

        # Wage YoY Change
        axes[0].set_ylim(-15, 15)
        #axes[0].set_title("Wage YoY Change (%)")
        axes[0].set_title("Wage YoY Change (%)")
        axes[0].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[0].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[0].set_ylabel("Percent Change")
        axes[0].legend()
        axes[0].grid(True)

        # Employment %
        axes[1].plot(occ_data['year'].to_numpy(), occ_data['Employment_Pct_Change'].to_numpy(), marker='o', color='darkorange')

        #sns.lineplot(ax=axes[1], data=occ_data, x='year', y='Employment_Pct_Change', marker='o', color='darkorange')
        axes[1].set_ylim(-15, 15)
        axes[1].set_title("Employment YoY Change (%)")
        axes[1].axvline(2020, color='gray', linestyle='--', label='COVID Start')
        axes[1].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
        axes[1].set_ylabel("Percent Change")
        axes[1].legend()
        axes[1].grid(True)

        for ax in axes:
            ax.set_xlabel("Year")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_title = occ_title.replace(' ', '_').replace('/', '_')
        fig_path = f"{directory}/yoy_pct_change_{safe_title}_{occ_code}.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"âœ… Saved: {fig_path}")




def cluster_occupations_by_yoy_trends_auto(data, nlp_name, directory=".", max_k=10):
    print("ðŸ” Starting clustering analysis for 2022â€“2024 YoY changes...")

    # Ensure correct types
    data['year'] = pd.to_numeric(data['year'], errors='coerce').astype('Int64')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )
   # data = data.dropna()

    print("âœ… Unique years in dataset:", sorted(data['year'].dropna().unique()))

    # Sort and compute YoY changes
    data = data.sort_values(by=['OCC_CODE', 'year'])
    data['Wage_Pct_Change'] = data.groupby('OCC_CODE')['a_mean_per_year'].pct_change() * 100
    data['Employment_Pct_Change'] = data.groupby('OCC_CODE')['tot_emp_per_year'].pct_change() * 100
    
    #recent_data = recent_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])


    # Filter for 2022â€“2024
    recent_data = data[data['year'].isin([2022, 2023, 2024])].copy()
    recent_data = recent_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

    print("ðŸ“Š Recent data shape:", recent_data.shape)

    # Pivot to create feature matrix
    #pivot_wage = recent_data.pivot(index='OCC_CODE', columns='year', values='Wage_Pct_Change')
    #pivot_emp = recent_data.pivot(index='OCC_CODE', columns='year', values='Employment_Pct_Change')

    pivot_wage = (recent_data.groupby(['OCC_CODE', 'year'])['Wage_Pct_Change'].mean().unstack())
    pivot_emp = (recent_data.groupby(['OCC_CODE', 'year'])['Employment_Pct_Change'].mean().unstack())



    features = pd.concat([pivot_wage, pivot_emp], axis=1)

    if features.shape[1] != 6:
        print(f"âŒ Feature matrix malformed: expected 6 columns, got {features.shape[1]}")
        return

    features.columns = ['wage_2022', 'wage_2023', 'wage_2024', 'emp_2022', 'emp_2023', 'emp_2024']
    features = features.dropna()

    # Merge metadata
    occ_info = data.drop_duplicates('OCC_CODE')[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
    merged = features.merge(occ_info, on='OCC_CODE', how='left').dropna()

    # Scale features
    X = merged[['wage_2022', 'wage_2023', 'wage_2024', 'emp_2022', 'emp_2023', 'emp_2024']]
    
    # Replace infinite values with NaN, then drop any rows with NaNs
    #X.replace([np.inf, -np.inf], np.nan, inplace=True)
    #X = X.dropna()
    
# Drop rows with inf or NaN across X columns and keep index aligned with merged
    valid_mask = np.isfinite(X).all(axis=1)
    X = X[valid_mask]
    merged = merged.loc[valid_mask].reset_index(drop=True)


    X_scaled = StandardScaler().fit_transform(X)

    # Silhouette-based cluster search
    silhouette_scores = {}
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores[k] = score
        print(f"ðŸ“ˆ Silhouette score for k={k}: {score:.3f}")

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"âœ… Optimal number of clusters: {optimal_k}")

    # Final KMeans clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    merged['Cluster'] = final_kmeans.fit_predict(X_scaled)

    # Plotting
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        merged[f'Average_Capability_Similarity_{nlp_name}_Percent'],
        merged['wage_2024'],
        c=merged['Cluster'],
        cmap='tab10',
        s=60,
        alpha=0.8
    )
    plt.xlabel("AI-OCI (%)")
    plt.ylabel("Wage YoY Change (2024)")
    plt.title("Clusters Based on YoY Wage & Employment Trends (2022â€“2024)")
    plt.colorbar(scatter, label="Cluster ID")
    plt.grid(True)
    plt.tight_layout()
    output_path = f"{directory}/clusters_yoy_2022_2024.pdf"
    plt.savefig(output_path)
    plt.show()

    print(f"âœ… Clustering completed and plot saved to: {output_path}")


def compute_6_point_relative_scores(data, nlp_name):
    """
    Computes 6-point relative change (Wage and Employment YoY % changes from 2022â€“2024)
    and returns a DataFrame with AI-OCI and Title.
    """
    # Ensure types
    data['year'] = pd.to_numeric(data['year'], errors='coerce').astype('Int64')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )

    # Compute YoY % changes
    data = data.sort_values(by=['OCC_CODE', 'year'])
    data['Wage_Pct_Change'] = data.groupby('OCC_CODE')['a_mean_per_year'].pct_change() * 100
    data['Employment_Pct_Change'] = data.groupby('OCC_CODE')['tot_emp_per_year'].pct_change() * 100

    # Filter for 2022â€“2024 and drop NAs
    recent_data = data[data['year'].isin([2022, 2023, 2024])].copy()
    recent_data = recent_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

    # Group and pivot
    wage_pivot = recent_data.groupby(['OCC_CODE', 'year'])['Wage_Pct_Change'].mean().unstack()
    emp_pivot = recent_data.groupby(['OCC_CODE', 'year'])['Employment_Pct_Change'].mean().unstack()
    features = pd.concat([wage_pivot, emp_pivot], axis=1)

    # Rename and drop NAs
    features.columns = ['wage_2022', 'wage_2023', 'wage_2024', 'emp_2022', 'emp_2023', 'emp_2024']
    features = features.dropna()

    # Compute the 6-point score
    features['Score'] = features.sum(axis=1)

    # Merge metadata
    occ_info = data.drop_duplicates('OCC_CODE')[['OCC_CODE', 'Title', f'Average_Capability_Similarity_{nlp_name}_Percent']]
    final = features.merge(occ_info, on='OCC_CODE', how='left').dropna()

    return final


def overlay_yoy_plots(
    data,
    nlp_name,
    selected_titles,
    directory=".",
    title_prefix="Overlay:"
):
    os.makedirs(directory, exist_ok=True)

    #data = pd.read_csv(data_path)
    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data['a_mean_per_year'] = pd.to_numeric(data['a_mean_per_year'], errors='coerce')
    data['tot_emp_per_year'] = pd.to_numeric(data['tot_emp_per_year'], errors='coerce')
    data[f'Average_Capability_Similarity_{nlp_name}_Percent'] = pd.to_numeric(
        data[f'Average_Capability_Similarity_{nlp_name}_Percent'], errors='coerce'
    )
    data = data.dropna(subset=['year', 'a_mean_per_year', 'tot_emp_per_year', f'Average_Capability_Similarity_{nlp_name}_Percent'])

    selected_data = data[data['Title'].isin(selected_titles)]
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    colors = sns.color_palette("tab10", len(selected_titles))

    for i, title in enumerate(selected_titles):
        occ_data = selected_data[selected_data['Title'] == title].sort_values('year')
        occ_data['Wage_Pct_Change'] = occ_data['a_mean_per_year'].pct_change() * 100
        occ_data['Employment_Pct_Change'] = occ_data['tot_emp_per_year'].pct_change() * 100
        occ_data = occ_data.dropna(subset=['Wage_Pct_Change', 'Employment_Pct_Change'])

        if occ_data.empty:
            continue

        axes[0].plot(occ_data['year'].to_numpy(), occ_data['Wage_Pct_Change'].to_numpy(), marker='o', label=title, color=colors[i])
        axes[1].plot(occ_data['year'].to_numpy(), occ_data['Employment_Pct_Change'].to_numpy(), marker='o', label=title, color=colors[i])

    axes[0].set_title("Wage YoY Change (%)", fontsize=10)
    axes[0].axvline(2020, color='gray', linestyle='--', label='COVID Start')
    axes[0].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
    axes[0].set_ylim(-20, 35)
    axes[0].set_ylabel("Percent Change",fontsize =15)
    #axes[0].legend()
    axes[0].grid(True)

    axes[1].set_title("Employment YoY Change (%)", fontsize =10)
    axes[1].axvline(2020, color='gray', linestyle='--', label='COVID Start')
    axes[1].axvline(2022, color='green', linestyle='--', label='LLM Adoption')
    axes[1].set_ylim(-30, 85)
    axes[1].set_ylabel("Percent Change", fontsize =15)
    #axes[1].legend()
    axes[1].legend(fontsize=6, loc='upper left')  # or inside ax[0] if more space

    axes[1].grid(True)

    for ax in axes:
        ax.set_xlabel("Year",fontsize = 15)
        ax.tick_params(axis='both', labelsize=10)
    
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #fig.suptitle(f'{title_prefix} Wage and Employment Changes for Selected Occupations', fontsize=14)
    fig_path = f"{directory}/overlay_yoy_selected_occupations.png"
    plt.savefig(fig_path, dpi=300)
    fig_path = f"{directory}/overlay_yoy_selected_occupations.pdf"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return fig_path



def plot_aioci_vs_combined_score(data, nlp_name, directory="."):
    print("ðŸ“ˆ Plotting AI-OCI vs Combined 6-Point Relative Change Score...")

    # Step 1: Compute relative change features and score
    merged = compute_6_point_relative_scores(data, nlp_name)
    print ('test',merged.head())


      # Clean invalid values
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged.dropna(subset=['Score', f'Average_Capability_Similarity_{nlp_name}_Percent'])
    
    print ('test2', merged.head())
    # Compute stats
    mean_score = merged['Score'].mean()
    std_score = merged['Score'].std()
    plus_1sd = mean_score + std_score
    minus_1sd = mean_score - std_score

    print(f"Mean Score: {mean_score:.2f}")
    print(f"+1 SD: {plus_1sd:.2f}")
    print(f"-1 SD: {minus_1sd:.2f}")

    # Step 2: Plotting
    plt.figure(figsize=(7,5.5))
    sns.scatterplot(
        data=merged,
        x='Score',
        y=f'Average_Capability_Similarity_{nlp_name}_Percent',
        hue='Score',
        palette='coolwarm',
        edgecolor='black',
        s=60,
        alpha=0.8,
        legend=False
    )

    # Add KDE
    sns.kdeplot(
        data=merged,
        x='Score',
        bw_adjust=1.5,
        fill=True,
        alpha=0.15,
        color='purple',
        label='KDE (Score Distribution)'
    )

    # Add vertical lines for mean and Â±1 SD
    plt.axvline(mean_score, color='blue', linestyle='--', label=f'Mean = {mean_score:.2f}')
    plt.axvline(minus_1sd, color='red', linestyle=':', label=f'-1 SD = {minus_1sd:.2f}')
    plt.axvline(plus_1sd, color='green', linestyle=':', label=f'+1 SD = {plus_1sd:.2f}')

    # Add axes lines
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

    # Annotate outliers
    for _, row in merged.iterrows():
        is_high_aioci = row[f'Average_Capability_Similarity_{nlp_name}_Percent'] > 30
        is_outlier = (row['Score'] > plus_1sd and row[f'Average_Capability_Similarity_{nlp_name}_Percent'] > 40) or (row['Score'] < minus_1sd)
        if is_high_aioci and is_outlier:
            plt.text(
                row['Score'] + 0.5,
                row[f'Average_Capability_Similarity_{nlp_name}_Percent'] + 0.5,
                row['OCC_CODE'],  # using code instead of title
                fontsize=10,
                color='black',
                weight='bold'
            )

    # Labels and limits
    plt.xlabel("6-Point Combined Relative Change Score (2022â€“2024)", fontsize = 15)
    plt.ylabel("AI-OCI", fontsize =15)
    #plt.title("AI-OCI vs. Combined Wage & Employment Change Score (2022â€“2024)")
    plt.xlim(merged['Score'].min(), 100)
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = f"{directory}/aioci_vs_combined_score_6points.pdf"
    plt.savefig(output_path)
    plt.close()
    print(f"âœ… Saved plot to: {output_path}")

    # Save CSV
    debug_df = merged[['OCC_CODE', 'Title', 'wage_2022', 'wage_2023', 'wage_2024',
                       'emp_2022', 'emp_2023', 'emp_2024', 'Score',
                       f'Average_Capability_Similarity_{nlp_name}_Percent']].copy()
    csv_path = f"{directory}/aioci_combined_score_debug.csv"
    debug_df.to_csv(csv_path, index=False)
    print(f"âœ… Debug CSV saved to: {csv_path}")
        # Get outlier titles for follow-up
    outliers = merged[
        (
            (merged['Score'] > mean_score + std_score) |
            (merged['Score'] < mean_score - std_score)
            )&
        (merged[f'Average_Capability_Similarity_{nlp_name}_Percent'] > 30)
    ]
   # outliers = ((outliers['Score'] > plus_1sd and outliers[f'Average_Capability_Similarity_{nlp_name}_Percent'] > 40) or (outliers['Score'] < minus_1sd))
    outliers = outliers[
    ((outliers['Score'] > plus_1sd) & (outliers[f'Average_Capability_Similarity_{nlp_name}_Percent'] > 40)) |
    (outliers['Score'] < minus_1sd)
]
    outlier_titles = outliers['Title'].unique().tolist()
    return outlier_titles  # New return


def generate_outlier_composite(data, nlp_name, directory="."):
    print("ðŸ§© Generating Outlier Composite Visualization...")

    # Plot AI-OCI vs Score (also returns list of outlier titles)
    outlier_titles = plot_aioci_vs_combined_score(data, nlp_name, directory)

    # Plot wage/employment overlays for outliers
    overlay_yoy_plots(
        data=data,
        nlp_name=nlp_name,
        selected_titles=outlier_titles,
        directory=directory,
        title_prefix="Outliers:"
    )

    print("âœ… Outlier composite complete.")


# Main function
def main(pca_percent, nlp):
    directory = '/nfs/student/f/fawuahgyasi/public_html/AI-OCI/results_2/Analysis_plots_3/'
    file_suffix = 'no_pca' if pca_percent is None else str(pca_percent)
    data_file = f'{directory}/np_full_data_clean_{nlp}_{file_suffix}.xlsx'
    
    print('Here is ',nlp)
    # Call process_and_save to prepare data
    process_and_save(pca_percent, nlp, directory)

    # Load data for further analysis
    data_nodate = pd.read_excel(data_file)
    
    # Include overtime data for year column
    overtime_df = pd.read_csv('DATA/combined_data.csv')
    overtime_maj_df = overtime_df[overtime_df['group'] == 'major']
    overtime_df = overtime_df.loc[overtime_df['group'] == 'detailed']

    # Clean numeric columns BEFORE the merge
    overtime_df['tot_emp_per_year'] = pd.to_numeric(
    overtime_df['tot_emp_per_year'].astype(str).str.replace(',', ''), errors='coerce'
)
    overtime_df['a_mean_per_year'] = pd.to_numeric(
    overtime_df['a_mean_per_year'].astype(str).str.replace(',', ''), errors='coerce'
)



    overtime_df = overtime_df[['OCC_CODE', 'a_mean_per_year', 'tot_emp_per_year', 'year']]
    data = data_nodate.merge(overtime_df, on='OCC_CODE', how='left')  # Merge to include year
    print('error1') 
    # Aggregate employment data by occupation and year
    grouped_data = data.groupby(['OCC_CODE', 'year'])['tot_emp_per_year'].mean().reset_index()
    print('Error2')
    # Merge the aggregated data back into the main DataFrame
    data = data.merge(grouped_data, on=['OCC_CODE', 'year'], how='left', suffixes=('', '_mean'))
    #a_mean_per_year
    # Create era-specific columns
    data['TOT_EMP_Pre_COVID'] = data.loc[data['year'] <= 2019, 'tot_emp_per_year_mean']
    data['TOT_EMP_COVID'] = data.loc[(data['year'] >= 2020) & (data['year'] <= 2021), 'tot_emp_per_year_mean']
    data['TOT_EMP_LLM'] = data.loc[data['year'] >= 2022, 'tot_emp_per_year_mean']

    print('befor',data)
    #--do not uncomment 
    #data['TOT_EMP_Pre_COVID'] = data.loc[data['year'] <= 2019, 'TOT_EMP']
    #data['TOT_EMP_COVID'] = data.loc[(data['year'] >= 2020) & (data['year'] <= 2021), 'TOT_EMP']
    #data['TOT_EMP_LLM'] = data.loc[data['year'] >= 2022, 'TOT_EMP']



   # Plot overall correlation for wages
    plot_overall_correlation(
        data=data,
        x_col=f'Average_Capability_Similarity_{nlp}_Percent',
        y_col='A_MEAN',
        title=f'Overall Correlation Between AI-OCI and Wages ({nlp})',
        x_label='AI-OCI Similarity Percent',
        y_label='Mean Annual Wages',
        filename=f'overall_correlation_wages_{nlp}.png',
        directory=directory
    )

    # Plot overall correlation for employment
    plot_overall_correlation(
        data=data,
        x_col=f'Average_Capability_Similarity_{nlp}_Percent',
        y_col='TOT_EMP',
        title=f'Overall Correlation Between AI-OCI and Employment ({nlp})',
        x_label='AI-OCI Similarity Percent',
        y_label='Total Employment',
        filename=f'overall_correlation_employment_{nlp}.png',
        directory=directory
    )

    # Process and plot data
    process_and_plot(data_nodate, nlp, directory, file_suffix)
    
    # Plot major occupations ranking
    plot_major_occupations_ranking(data_nodate, nlp, directory)
    
    # Merge with AIOE data and compute correlation
    aioe_file = 'DATA/AIOE_csv.csv'
    merge_and_compute_aioe_correlation(data, aioe_file, nlp, directory, file_suffix)

    # Plot correlation by era for wages and employment
    plot_correlation_by_era(data, nlp, directory)
    #print('plots should be done') 
    
    # plot correlation by era for wages and employment
    plot_combined_correlation_by_era(data, nlp, directory) 
    
    # plot relative change in employment
    

    plot_relative_change_grouped_bar(overtime_maj_df, directory)
    plot_relative_change_grouped_bar_with_error(overtime_maj_df, directory)
    plot_top_aioci_occupation_trends(data, nlp, directory, top_n=40)
    plot_percentage_change_for_top_occupations(
    data=data,                         # merged detailed-level data with year
    nlp_name=nlp,                      # selected NLP model name
    top_n=10,                          # number of top occupations to plot
    directory=directory,              # output directory
    start_year=2012                   # base year for calculating percentage change
)

    plot_percentage_change_for_bottom_occupations(
    data=data,
    nlp_name=nlp,
    bottom_n=10,
    directory=directory,
    start_year=2012
)
    plot_yoy_change_for_bottom_occupations(
    data, 
    nlp_name=nlp, 
    bottom_n=10, 
    directory=directory, 
    title_prefix="Bottom 10: "
)

    plot_yoy_change_for_top_occupations(data=data,nlp_name=nlp,top_n=30,
    directory=directory,
    title_prefix="Top 30: "
)


    cluster_occupations_by_yoy_trends_auto(data=data, nlp_name=nlp, directory=directory)
    plot_aioci_vs_combined_score(data=data, nlp_name=nlp, directory=directory)

    generate_outlier_composite(data=data, nlp_name=nlp, directory=directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCA files and generate embeddings.")
    parser.add_argument('--pca_percent', type=float, help="PCA percentage to be used for the embeddings.")
    parser.add_argument('--nlp', type=int, required=True, choices=[1, 2, 3, 4, 5], help="NLP model selection: 1 for BERT, 2 for OpenAI, 3 for OpenAI_LG, 4 for OpenAI_SM, 5 for E5")

    args = parser.parse_args()

    nlp_dict = {1: 'bert', 2: 'openai', 3: 'openai_lg', 4: 'openai_sm', 5: 'e5'}
    selected_nlp = nlp_dict[args.nlp]

    main(args.pca_percent, selected_nlp)

    print("Process completed successfully.")
