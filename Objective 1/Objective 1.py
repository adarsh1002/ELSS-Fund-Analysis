# %%


# %%


# %% [markdown]
# NAV Growth Analysis

# %%

import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = 'Sampled_perf_direct.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert 'NAV Date' to datetime format
df['NAV Date'] = pd.to_datetime(df['NAV Date'])

# Filter necessary columns
df_nav = df[['Scheme Name', 'NAV Date', 'NAV Direct']]

# Pivot the data: Rows = NAV Date, Columns = Scheme Name, Values = NAV Direct
df_pivot = df_nav.pivot_table(index='NAV Date', columns='Scheme Name', values='NAV Direct')

# Index each scheme's NAV to 100 at start
df_indexed = df_pivot.divide(df_pivot.iloc[0]).multiply(100)

# Plot the trend line
plt.figure(figsize=(14,8))
for scheme in df_indexed.columns:
    plt.plot(df_indexed.index, df_indexed[scheme], label=scheme, linewidth=4)

plt.title('Indexed NAV Trend of ELSS Mutual Funds (Jan 2020 - Dec 2024)', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Indexed NAV (Base = 100)', fontsize=14)
plt.grid(True, linestyle='--', alpha=1)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# NAV Growth Analysis Plotly Graph - Interactive

# %%
import pandas as pd
import plotly.graph_objects as go

# Load the data
file_path = 'Sampled_perf_direct.xlsx'

df = pd.read_excel(file_path, sheet_name='Sheet1')

# Convert 'NAV Date' to datetime
df['NAV Date'] = pd.to_datetime(df['NAV Date'])

# Pivot the data: NAV Date as index, Scheme Name as columns
df_nav = df[['Scheme Name', 'NAV Date', 'NAV Direct']]
df_pivot = df_nav.pivot_table(index='NAV Date', columns='Scheme Name', values='NAV Direct')

# Index each fund's NAV to 100 at start
df_indexed = df_pivot.divide(df_pivot.iloc[0]).multiply(100)

# Create the Plotly Figure
fig = go.Figure()

# Add a line for each fund
for scheme in df_indexed.columns:
    fig.add_trace(go.Scatter(
        x=df_indexed.index,
        y=df_indexed[scheme],
        mode='lines',
        name=scheme,
        hovertemplate='%{y:.2f} (Indexed NAV) on %{x|%b %Y}<extra>%{fullData.name}</extra>'
    ))

# Layout customization
fig.update_layout(
    title='Interactive Indexed NAV Trend of ELSS Mutual Funds (2020-2024)',
    xaxis_title='Date',
    yaxis_title='Indexed NAV (Base = 100)',
    hovermode='x unified',
    legend_title='Scheme Name',
    template='plotly_white',
    height=600,
    width=1000
)

fig.show()


# %%


# %%
import pandas as pd
import plotly.express as px

# Load the file
file_path = 'Sampled_perf_direct.xlsx'
df = pd.read_excel(file_path)

# Prepare data
df['NAV Date'] = pd.to_datetime(df['NAV Date'])  # Convert to datetime
df = df.rename(columns={'Daily AUM (Cr.)': 'AUM'})  # Rename for ease

df["AUM"]=df['AUM'].str.replace(",","")
df["AUM"]=df['AUM'].str.replace("^","")
df['AUM']=pd.to_numeric(df['AUM'])
df.info()
df.head()
df.to_excel("Performance data_main.xlsx")

# %%


# %%




# %% [markdown]
# AUM Growth Analysis

# %%
import pandas as pd
import plotly.express as px

# Load the file
file_path = 'Sampled_perf_direct.xlsx'
df = pd.read_excel(file_path)
df.head()
# Prepare data
df['NAV Date'] = pd.to_datetime(df['NAV Date'])  # Convert to datetime
df = df.rename(columns={'Daily AUM (Cr.)': 'AUM'})  # Rename for ease

df['AUM'] = df['AUM'].astype("str").str.strip()
df["AUM"]=df['AUM'].str.replace("^","")
df["AUM"]=df['AUM'].str.replace(",","")
df['AUM'] = pd.to_numeric(df['AUM'])
# Sort for clean plotting
aum_df = df[['Scheme Name', 'NAV Date', 'AUM']].dropna()
aum_df = aum_df.sort_values(by=['Scheme Name', 'NAV Date'])
schemes = aum_df['Scheme Name'].unique()

# Step 1: Find last AUM value for each scheme
last_aum = aum_df.sort_values('NAV Date').groupby('Scheme Name').tail(1)
last_aum_sorted = last_aum.sort_values(by='AUM', ascending=False)
sorted_schemes = last_aum_sorted['Scheme Name'].values

# Step 2: Now plot based on sorted_schemes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
axes = axes.flatten()

for i, focus_scheme in enumerate(sorted_schemes):
    ax = axes[i]
    # Plot all schemes in light background
    for scheme in sorted_schemes:
        scheme_data = aum_df[aum_df['Scheme Name'] == scheme]
        ax.plot(
            scheme_data['NAV Date'],
            scheme_data['AUM'],
            color='gray',
            linewidth=1.2,
            alpha=0.7
        )
    # Highlight focused scheme
    focus_data = aum_df[aum_df['Scheme Name'] == focus_scheme]
    ax.plot(
        focus_data['NAV Date'],
        focus_data['AUM'],
        color='blue',
        linewidth=3,
        label=focus_scheme
    )
    
    ax.set_title(focus_scheme, fontsize=11)
    ax.set_xlabel('Date')
    ax.set_ylabel('AUM (Cr.)')
    ax.grid(True, linestyle='--', alpha=0.5)

# Remove extra subplots if any
for j in range(len(sorted_schemes), len(axes)):
    fig.delaxes(axes[j])

# Main Title
fig.suptitle('AUM Growth of Top 5 ELSS Mutual Funds (Panels Sorted by Last AUM Size)', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [markdown]
# Rolling returns analysis Fund wise for 1, 3 and 5 years

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded file
file_path = 'Sampled_perf_direct.xlsx'
df = pd.read_excel(file_path)

# Prepare Data
df['NAV Date'] = pd.to_datetime(df['NAV Date'])
df = df.rename(columns={
    'Return 1 Year (%) Direct': 'Return_1Y',
    'Return 3 Year (%) Direct': 'Return_3Y',
    'Return 5 Year (%) Direct': 'Return_5Y'
})

# Convert returns to numeric
for col in ['Return_1Y', 'Return_3Y', 'Return_5Y']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare working dataframe
returns_df = df[['Scheme Name', 'NAV Date', 'Return_1Y', 'Return_3Y', 'Return_5Y']].dropna()
returns_df = returns_df.sort_values(by=['Scheme Name', 'NAV Date'])

schemes = returns_df['Scheme Name'].unique()

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 12))
axes = axes.flatten()

smooth_window = 3  # 3 months rolling mean

for i, scheme in enumerate(schemes):
    ax = axes[i]
    
    scheme_data = returns_df[returns_df['Scheme Name'] == scheme].copy()
    scheme_data = scheme_data.set_index('NAV Date')
    
    # Only take Return columns
    scheme_returns = scheme_data[['Return_1Y', 'Return_3Y', 'Return_5Y']]
    
    # Rolling mean smoothing
    scheme_smoothed = scheme_returns.rolling(window=smooth_window, min_periods=1).mean()
    
    # Plot
    ax.plot(scheme_smoothed.index, scheme_smoothed['Return_1Y'], label='1-Year Return', color='blue', linewidth=1.5)
    ax.plot(scheme_smoothed.index, scheme_smoothed['Return_3Y'], label='3-Year Return', color='green', linewidth=1.5)
    ax.plot(scheme_smoothed.index, scheme_smoothed['Return_5Y'], label='5-Year Return', color='red', linewidth=1.5)
    
    ax.set_title(scheme, fontsize=11)
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns (%)')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=8)

# Remove extra subplots
for j in range(len(schemes), len(axes)):
    fig.delaxes(axes[j])

# Overall Title
fig.suptitle('Smoothed Trend Analysis of 1-Year, 3-Year, 5-Year Returns of Top ELSS Funds', fontsize=18)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %% [markdown]
# Rolling returns analysis for different funds accross different horizons

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
file_path = "Sampled_perf_direct.xlsx"
df = pd.read_excel(file_path)

# Rename and clean
df['NAV Date'] = pd.to_datetime(df['NAV Date'])
df = df.rename(columns={
    'Return 1 Year (%) Direct': 'Return_1Y',
    'Return 3 Year (%) Direct': 'Return_3Y',
    'Return 5 Year (%) Direct': 'Return_5Y'
})

# Convert returns to numeric
for col in ['Return_1Y', 'Return_3Y', 'Return_5Y']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare working dataframe
returns_df = df[['Scheme Name', 'NAV Date', 'Return_1Y', 'Return_3Y', 'Return_5Y']].dropna()
returns_df = returns_df.sort_values(by=['NAV Date', 'Scheme Name'])

# Unique schemes list
schemes = returns_df['Scheme Name'].unique()
colors = plt.cm.tab10.colors  # Up to 10 different colors for schemes

# Plotting Facet Grids
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(16, 15), sharex=True)

returns = ['Return_1Y', 'Return_3Y', 'Return_5Y']
titles = ['1-Year Returns (%)', '3-Year Returns (%)', '5-Year Returns (%)']

for idx, ret in enumerate(returns):
    ax = axes[idx]
    for i, scheme in enumerate(schemes):
        scheme_data = returns_df[returns_df['Scheme Name'] == scheme]
        ax.plot(scheme_data['NAV Date'], scheme_data[ret],
                label=scheme, color=colors[i % len(colors)], linewidth=2)
        
    ax.set_title(titles[idx], fontsize=14)
    ax.set_ylabel('Returns (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Date', fontsize=14)

# Legend outside the plot
fig.legend(schemes, loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=len(schemes), fontsize=12)

fig.suptitle('Returns Trend of ELSS Funds (Different Panels for 1Y, 3Y, 5Y)', fontsize=20)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()


# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load and prepare data
file_path = "Sampled_perf_direct.xlsx"
df = pd.read_excel(file_path)

# Rename and clean
df['NAV Date'] = pd.to_datetime(df['NAV Date'])
df = df.rename(columns={
    'Return 1 Year (%) Direct': 'Return_1Y',
    'Return 3 Year (%) Direct': 'Return_3Y',
    'Return 5 Year (%) Direct': 'Return_5Y'
})

# Convert returns to numeric
for col in ['Return_1Y', 'Return_3Y', 'Return_5Y']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Prepare working dataframe
returns_df = df[['Scheme Name', 'NAV Date', 'Return_1Y', 'Return_3Y', 'Return_5Y']].dropna()
returns_df = returns_df.sort_values(by=['NAV Date', 'Scheme Name'])

# Unique schemes list
schemes = returns_df['Scheme Name'].unique()
colors = plt.cm.tab10.colors  # Up to 10 different colors for schemes

# Plotting Facet Grids
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 18), sharex=True)

returns = ['Return_1Y', 'Return_3Y', 'Return_5Y']
titles = ['1-Year Returns (%)', '3-Year Returns (%)', '5-Year Returns (%)']

for idx, ret in enumerate(returns):
    ax = axes[idx]
    for i, scheme in enumerate(schemes):
        scheme_data = returns_df[returns_df['Scheme Name'] == scheme]
        
        # Plot full line
        ax.plot(scheme_data['NAV Date'], scheme_data[ret],
                label=scheme, color=colors[i % len(colors)], linewidth=2)
        
        # Annotate points: pick 1 point per year (January of each year ideally)
        for year in range(2021, 2025):  # 2021-2024
            point = scheme_data[(scheme_data['NAV Date'].dt.year == year)]
            if not point.empty:
                nav_date = point['NAV Date'].iloc[0]
                return_value = point[ret].iloc[0]
                ax.annotate(f"{return_value:.1f}%", 
                            xy=(nav_date, return_value),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color=colors[i % len(colors)])

    ax.set_title(titles[idx], fontsize=14)
    ax.set_ylabel('Returns (%)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

axes[-1].set_xlabel('Date', fontsize=14)

# Legend
fig.legend(schemes, loc='upper center', bbox_to_anchor=(0.5, -0.03),
           ncol=len(schemes), fontsize=12)

fig.suptitle('Annotated Returns Trend of ELSS Funds (Panels for 1Y, 3Y, 5Y)', fontsize=20)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()


# %% [markdown]
# Returns Heatmap Analysis

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your data
file_path = 'Performance data_main.xlsx'
df = pd.read_excel(file_path)

# Prepare data
df['NAV Date'] = pd.to_datetime(df['NAV Date'])
df = df.rename(columns={
    'Return 1 Year (%) Direct': 'Return_1Y',
    'Return 3 Year (%) Direct': 'Return_3Y',
    'Return 5 Year (%) Direct': 'Return_5Y'
})

# Selecting required columns
df = df[['Scheme Name', 'NAV Date', 'Return_1Y', 'Return_3Y', 'Return_5Y', 'AUM']]

# Group by Scheme and calculate averages
avg_df = df.groupby('Scheme Name').agg({
    'Return_1Y': 'mean',
    'Return_3Y': 'mean',
    'Return_5Y': 'mean',
    'AUM': 'max'
}).reset_index()

# Separate returns and AUM
returns_df = avg_df[['Scheme Name', 'Return_1Y', 'Return_3Y', 'Return_5Y']]
returns_df.set_index('Scheme Name', inplace=True)

aum_series = avg_df.set_index('Scheme Name')['AUM']

# Expand AUM to match returns_df structure
aum_background = pd.DataFrame(
    np.tile(aum_series.values[:, None], (1, returns_df.shape[1])),
    index=returns_df.index,
    columns=returns_df.columns
)
returns_with_percent = returns_df.round(1).astype(str) + '%'

# Plotting
plt.figure(figsize=(12, 6))
sns.heatmap(
    aum_background,  # Use actual AUM for color intensity
    annot=returns_with_percent, fmt='', linewidths=0.5,
    annot_kws={ "size":16},
    cmap=sns.light_palette("seagreen", as_cmap=True),
    cbar_kws={'label': 'AUM Size (₹ Cr)'},
    yticklabels=returns_df.index
)

plt.title('Heatmap of Average Returns (Cell Color Coded by AUM Size)', fontsize=18)
plt.xlabel('Return Period', fontsize=14)
plt.ylabel('Scheme Name', fontsize=14)
plt.tight_layout()
plt.show()


