import json, pickle, warnings, os
warnings.filterwarnings('ignore')
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(__file__).resolve().parent / 'ml_models'

# ── Data loader ───────────────────────────────────────────────────────
def load_data():
    df   = pd.read_csv(BASE / 'Dairy_Demand_Intelligence_Dataset_2021_2023.csv', encoding='latin1')
    df_m = pd.read_csv(BASE / 'updated_Monthly_dataset.csv', encoding='latin1')
    df_w = pd.read_csv(BASE / 'updated_Weekly_dataset.csv', encoding='latin1')
    df['Date'] = pd.to_datetime(df['Date'])
    df_m['Year_Month'] = pd.to_datetime(df_m['Year_Month'])
    df_w['Week_Start']  = pd.to_datetime(df_w['Week_Start'])
    df['Revenue']   = df['Quantity Sold (Units)'] * df['Price (INR)']
    df_m['Revenue'] = df_m['Quantity_Sold'] * df_m['Avg_Price']
    season_map = {
        'January':'Winter','February':'Winter','March':'Spring','April':'Spring','May':'Spring',
        'June':'Monsoon','July':'Monsoon','August':'Monsoon','September':'Monsoon',
        'October':'Autumn','November':'Autumn','December':'Winter'
    }
    df['Season'] = df['Month'].map(season_map)
    return df, df_m, df_w

# ── Main dashboard ────────────────────────────────────────────────────
def index(request):
    df, df_m, df_w = load_data()

    total_units   = int(df['Quantity Sold (Units)'].sum())
    total_revenue = float(df['Revenue'].sum())
    avg_price     = float(df['Price (INR)'].mean())
    fest_avg   = float(df[df['Festival Indicator']=='Yes']['Quantity Sold (Units)'].mean())
    normal_avg = float(df[df['Festival Indicator']=='No']['Quantity Sold (Units)'].mean())
    fest_uplift = (fest_avg / normal_avg - 1) * 100
    yoy = df.groupby('Year')['Quantity Sold (Units)'].sum()
    growth_22 = float((yoy[2022]/yoy[2021]-1)*100)
    growth_23 = float((yoy[2023]/yoy[2022]-1)*100)

    monthly = df_m.groupby('Year_Month').agg(Units=('Quantity_Sold','sum'), Revenue=('Revenue','sum')).reset_index()
    prod    = df.groupby('Product Name').agg(Units=('Quantity Sold (Units)','sum'), Revenue=('Revenue','sum')).sort_values('Units', ascending=False).reset_index()
    reg     = df.groupby('Region').agg(Units=('Quantity Sold (Units)','sum'), Revenue=('Revenue','sum')).sort_values('Units', ascending=False).reset_index()
    top_fest = (df[df['Festival Name'].notna() & (df['Festival Name'] != '')]
                .groupby('Festival Name')['Quantity Sold (Units)'].mean()
                .sort_values(ascending=False).head(8).reset_index())
    season_order = ['Winter','Spring','Monsoon','Autumn']
    season  = df.groupby('Season')['Quantity Sold (Units)'].mean().reindex(season_order).reset_index()
    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    dow     = df.groupby('Day of Week')['Quantity Sold (Units)'].mean().reindex(dow_order).reset_index()
    states  = df.groupby('State')['Quantity Sold (Units)'].sum().sort_values(ascending=False).head(10).reset_index()
    pr_heat = df.pivot_table(values='Quantity Sold (Units)', index='Product Name', columns='Region', aggfunc='sum').fillna(0)
    temp_m  = df.groupby(df['Date'].dt.to_period('M')).agg(Units=('Quantity Sold (Units)','sum'), Temp=('Temperature (øC)','mean')).reset_index()
    yoy_df  = df.groupby('Year').agg(Units=('Quantity Sold (Units)','sum'), Revenue=('Revenue','sum')).reset_index()
    top3    = prod['Product Name'].head(3).tolist()
    weekly_prod = {}
    for p in top3:
        wts = df_w[df_w['Product_Name']==p].groupby('Week_Start')['Quantity_Sold'].sum().reset_index()
        weekly_prod[p] = {'labels': [d.strftime('%d %b %y') for d in wts['Week_Start']], 'values': [int(x) for x in wts['Quantity_Sold']]}

    products_list = sorted(df['Product Name'].unique().tolist())
    regions_list  = sorted(df['Region'].unique().tolist())

    ctx = {
        'kpis': {
            'total_units': f"{total_units:,}",
            'total_revenue': f"\u20b9{total_revenue/1e7:.1f} Cr",
            'avg_price': f"\u20b9{avg_price:.0f}",
            'fest_uplift': f"+{fest_uplift:.1f}%",
            'growth_22': f"{growth_22:+.1f}%",
            'growth_23': f"{growth_23:+.1f}%",
            'num_products': int(df['Product Name'].nunique()),
            'num_states': int(df['State'].nunique()),
        },
        'products_list': products_list,
        'regions_list':  regions_list,
        'chart_data': json.dumps({
            'monthly':  {'labels': [d.strftime('%b %Y') for d in monthly['Year_Month']], 'units': [int(x) for x in monthly['Units']], 'revenue': [round(float(x)/1e6,2) for x in monthly['Revenue']]},
            'products': {'labels': prod['Product Name'].tolist(), 'units': [int(x) for x in prod['Units']], 'revenue': [round(float(x)/1e6,2) for x in prod['Revenue']]},
            'regions':  {'labels': reg['Region'].tolist(), 'units': [int(x) for x in reg['Units']], 'revenue': [round(float(x)/1e6,2) for x in reg['Revenue']]},
            'festival': {'labels': top_fest['Festival Name'].tolist(), 'values': [round(float(x),1) for x in top_fest['Quantity Sold (Units)']], 'normal_avg': round(normal_avg,1)},
            'season':   {'labels': [str(s) for s in season['Season'].tolist()], 'values': [round(float(x),1) if not pd.isna(x) else 0 for x in season['Quantity Sold (Units)']]},
            'dow':      {'labels': dow_order, 'values': [round(float(x),1) if not pd.isna(x) else 0 for x in dow['Quantity Sold (Units)']]},
            'states':   {'labels': states['State'].tolist(), 'values': [int(x) for x in states['Quantity Sold (Units)']]},
            'heatmap':  {'products': pr_heat.index.tolist(), 'regions': pr_heat.columns.tolist(), 'values': [[int(v) for v in row] for row in pr_heat.values.tolist()]},
            'temp':     {'labels': [str(p) for p in temp_m['Date']], 'units': [int(x) for x in temp_m['Units']], 'temps': [round(float(x),1) for x in temp_m['Temp']]},
            'yoy':      {'years': [str(y) for y in yoy_df['Year']], 'units': [int(x) for x in yoy_df['Units']], 'revenue': [round(float(x)/1e6,2) for x in yoy_df['Revenue']]},
            'weekly_prod': weekly_prod,
            'products_list': products_list,
            'regions_list':  regions_list,
        })
    }
    return render(request, 'dashboard/index.html', ctx)


# ── FEATURE 1: Live Forecast API ──────────────────────────────────────
@csrf_exempt
def forecast_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    try:
        body    = json.loads(request.body)
        product = body.get('product', '')
        region  = body.get('region', '')
        months  = int(body.get('months', 3))

        key = f"{product}___{region}".replace(' ','_').replace('/','_').replace('(','').replace(')','')
        model_path = MODELS_DIR / f"{key}.pkl"

        if not model_path.exists():
            return JsonResponse({'error': f'Model not found for {product} / {region}'}, status=404)

        with open(model_path, 'rb') as f:
            saved = pickle.load(f)

        fit       = saved['model']
        history   = saved['history']
        last_date = saved['last_date']

        fcast    = fit.forecast(steps=months)
        fcast_ci = fit.get_forecast(steps=months).conf_int()

        # Build historical last 12 months
        hist_12 = history.tail(12)

        result = {
            'product': product,
            'region':  region,
            'history': {
                'labels': [str(d.strftime('%b %Y')) for d in hist_12.index],
                'values': [round(float(v)) for v in hist_12.values]
            },
            'forecast': {
                'labels': [str(d.strftime('%b %Y')) for d in fcast.index],
                'values': [max(0, round(float(v))) for v in fcast.values],
                'lower':  [max(0, round(float(v))) for v in fcast_ci.iloc[:,0].values],
                'upper':  [max(0, round(float(v))) for v in fcast_ci.iloc[:,1].values],
            },
            'summary': {
                'next_month':    max(0, round(float(fcast.values[0]))),
                'avg_forecast':  max(0, round(float(fcast.values.mean()))),
                'trend':         'increasing' if fcast.values[-1] > fcast.values[0] else 'decreasing',
                'last_actual':   round(float(history.values[-1])),
                'change_pct':    round((float(fcast.values[0]) / float(history.values[-1]) - 1)*100, 1)
            }
        }
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# ── FEATURE 2: AI Chatbot API ─────────────────────────────────────────
@csrf_exempt
def chatbot_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)
    try:
        body     = json.loads(request.body)
        question = body.get('question', '').strip()
        if not question:
            return JsonResponse({'error': 'Empty question'}, status=400)

        df, df_m, df_w = load_data()

        # Build rich dataset context
        product_sales = df.groupby('Product Name')['Quantity Sold (Units)'].sum().sort_values(ascending=False)
        region_sales  = df.groupby('Region')['Quantity Sold (Units)'].sum().sort_values(ascending=False)
        yoy           = df.groupby('Year')['Quantity Sold (Units)'].sum()
        fest_avg      = df[df['Festival Indicator']=='Yes']['Quantity Sold (Units)'].mean()
        normal_avg    = df[df['Festival Indicator']=='No']['Quantity Sold (Units)'].mean()
        season_avg    = df.groupby('Season')['Quantity Sold (Units)'].mean().sort_values(ascending=False)
        top_states    = df.groupby('State')['Quantity Sold (Units)'].sum().sort_values(ascending=False).head(5)
        top_festivals = (df[df['Festival Name'].notna() & (df['Festival Name']!='')]
                         .groupby('Festival Name')['Quantity Sold (Units)'].mean()
                         .sort_values(ascending=False).head(5))
        df['Revenue'] = df['Quantity Sold (Units)'] * df['Price (INR)']
        prod_rev      = df.groupby('Product Name')['Revenue'].sum().sort_values(ascending=False)
        price_by_prod = df.groupby('Product Name')['Price (INR)'].mean().round(2)

        context = f"""
You are an AI analyst for the SmartDairy Demand Intelligence System (SDDIS).
You have access to 3 years of Indian dairy sales data (2021-2023), 50,000 records, 12 products, 5 regions, 19 states.

KEY DATA FACTS:
- Total units sold: {int(df['Quantity Sold (Units)'].sum()):,}
- Total revenue: ₹{df['Revenue'].sum()/1e7:.1f} Crore
- Date range: Jan 2021 to Dec 2023
- Products: {', '.join(df['Product Name'].unique().tolist())}
- Regions: Central, East, North, South, West
- States: {', '.join(df['State'].unique().tolist())}

PRODUCT SALES RANKING (units):
{chr(10).join([f"  {i+1}. {p}: {int(v):,} units" for i,(p,v) in enumerate(product_sales.items())])}

PRODUCT REVENUE RANKING:
{chr(10).join([f"  {i+1}. {p}: ₹{v/1e6:.1f}M" for i,(p,v) in enumerate(prod_rev.items())])}

AVERAGE PRICES:
{chr(10).join([f"  {p}: ₹{v}" for p,v in price_by_prod.items()])}

REGIONAL SALES:
{chr(10).join([f"  {r}: {int(v):,} units" for r,v in region_sales.items()])}

YEAR-OVER-YEAR:
  2021: {int(yoy[2021]):,} units
  2022: {int(yoy[2022]):,} units  ({(yoy[2022]/yoy[2021]-1)*100:+.1f}% growth)
  2023: {int(yoy[2023]):,} units  ({(yoy[2023]/yoy[2022]-1)*100:+.1f}% growth)

SEASONAL PATTERNS:
{chr(10).join([f"  {s}: {v:.1f} avg units/entry" for s,v in season_avg.items()])}

FESTIVAL IMPACT:
  Festival days avg: {fest_avg:.1f} units
  Normal days avg:   {normal_avg:.1f} units
  Uplift: {(fest_avg/normal_avg-1)*100:.1f}%

TOP FESTIVALS BY SALES UPLIFT:
{chr(10).join([f"  {f}: {v:.1f} avg units" for f,v in top_festivals.items()])}

TOP 5 STATES:
{chr(10).join([f"  {s}: {int(v):,} units" for s,v in top_states.items()])}

FORECASTING MODELS USED: ARIMA, SARIMA (best - MAPE ~8.2%), Facebook Prophet, LSTM
BEST MODEL: SARIMA(1,1,1)(1,1,1,12) — captures annual seasonality best

Answer the user's question using this data. Be specific, use numbers, give actionable business insights.
Keep answer concise (3-5 sentences max unless asked for detail). Use ₹ for currency, bullet points when listing.
"""

        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "system": context,
            "messages": [{"role": "user", "content": question}]
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.anthropic.com/v1/messages',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'anthropic-version': '2023-06-01',
                'x-api-key': os.environ.get('ANTHROPIC_API_KEY', '')
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            answer = result['content'][0]['text']

        return JsonResponse({'answer': answer, 'question': question})

    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        # Fallback: rule-based answers if no API key
        answer = rule_based_answer(question, df, df_m)
        return JsonResponse({'answer': answer, 'question': question, 'mode': 'offline'})
    except Exception as e:
        answer = rule_based_answer(question, df, df_m)
        return JsonResponse({'answer': answer, 'question': question, 'mode': 'offline'})


def rule_based_answer(question, df, df_m):
    """Fallback rule-based answers when API key not set."""
    q = question.lower()
    df['Revenue'] = df['Quantity Sold (Units)'] * df['Price (INR)']

    if any(w in q for w in ['best', 'top', 'highest', 'most sold', 'popular']):
        if 'product' in q or 'sell' in q or 'sold' in q:
            top = df.groupby('Product Name')['Quantity Sold (Units)'].sum().idxmax()
            val = df.groupby('Product Name')['Quantity Sold (Units)'].sum().max()
            return f"**{top}** is the best-selling product with **{int(val):,} units** sold over 2021–2023. It dominates across all regions, especially Central and North."
        if 'region' in q:
            top = df.groupby('Region')['Quantity Sold (Units)'].sum().idxmax()
            val = df.groupby('Region')['Quantity Sold (Units)'].sum().max()
            return f"**{top} region** has the highest dairy demand with **{int(val):,} units** sold. This region should be prioritized for cold storage and distribution investments."
        if 'state' in q:
            top = df.groupby('State')['Quantity Sold (Units)'].sum().idxmax()
            return f"**{top}** is the top state by dairy demand. Focus distribution and promotional activities here for maximum ROI."
        if 'festival' in q:
            top = df[df['Festival Name'].notna() & (df['Festival Name']!='')].groupby('Festival Name')['Quantity Sold (Units)'].mean().idxmax()
            return f"**{top}** drives the highest dairy sales among all festivals. Stock up at least 2 weeks in advance to meet demand."

    if any(w in q for w in ['festival', 'diwali', 'holi', 'impact', 'uplift']):
        fest = df[df['Festival Indicator']=='Yes']['Quantity Sold (Units)'].mean()
        norm = df[df['Festival Indicator']=='No']['Quantity Sold (Units)'].mean()
        uplift = (fest/norm-1)*100
        return f"Festival days show a **{uplift:.1f}% uplift** in dairy sales (avg {fest:.0f} units vs {norm:.0f} on normal days). Top festivals: Diwali, Republic Day, Holi. Plan production **2 weeks ahead** of major festivals."

    if any(w in q for w in ['winter', 'summer', 'monsoon', 'season', 'seasonal']):
        season_avg = df.groupby('Season')['Quantity Sold (Units)'].mean().sort_values(ascending=False)
        best_s = season_avg.index[0]
        return f"**{best_s}** is the peak demand season for dairy. {season_avg.to_dict()}. Increase milk & butter inventory by 15-20% in Winter months (Oct–Feb)."

    if any(w in q for w in ['revenue', 'money', 'earn', 'profit', 'income']):
        total = df['Revenue'].sum()
        top_prod = df.groupby('Product Name')['Revenue'].sum().idxmax()
        return f"Total revenue over 2021–2023 is **₹{total/1e7:.1f} Crore**. **{top_prod}** generates the highest revenue due to its premium price point."

    if any(w in q for w in ['growth', 'yoy', 'year', '2021', '2022', '2023', 'trend']):
        yoy = df.groupby('Year')['Quantity Sold (Units)'].sum()
        g22 = (yoy[2022]/yoy[2021]-1)*100
        g23 = (yoy[2023]/yoy[2022]-1)*100
        return f"YoY Growth: 2021→2022: **{g22:+.1f}%**, 2022→2023: **{g23:+.1f}%**. The dairy market shows consistent growth — {int(yoy[2023]):,} units in 2023 vs {int(yoy[2021]):,} in 2021."

    if any(w in q for w in ['model', 'forecast', 'predict', 'arima', 'sarima', 'lstm', 'prophet', 'accuracy', 'mape']):
        return "Four models were tested — **SARIMA** achieved the best performance with **~8.2% MAPE**. Use SARIMA for monthly production planning, Prophet for festival-adjusted forecasts, and LSTM for real-time weekly demand adjustments."

    if any(w in q for w in ['temperature', 'weather', 'temp', 'hot', 'cold']):
        return "Temperature strongly influences dairy demand. **Milk demand peaks in Winter (15–18°C)** while buttermilk & ice cream see 30-40% higher sales when temperature exceeds 35°C in summer. Use weather forecasts as inputs to improve prediction accuracy."

    if any(w in q for w in ['ghee', 'butter', 'milk', 'curd', 'paneer', 'ice cream', 'lassi', 'cheese']):
        prod_name = next((p for p in df['Product Name'].unique() if p.lower() in q), None)
        if prod_name:
            units = int(df[df['Product Name']==prod_name]['Quantity Sold (Units)'].sum())
            rev   = df[df['Product Name']==prod_name]['Revenue'].sum()
            price = df[df['Product Name']==prod_name]['Price (INR)'].mean()
            return f"**{prod_name}**: {units:,} total units sold | Revenue: ₹{rev/1e6:.1f}M | Avg price: ₹{price:.0f}. Check the Products section for detailed weekly and regional breakdown."

    if any(w in q for w in ['recommend', 'suggest', 'advice', 'should', 'improve', 'optimize']):
        return ("Key recommendations based on data analysis:\n"
                "• **Increase inventory by 20%** in Oct–Dec for winter peak demand\n"
                "• **Stock up 2 weeks before** Diwali, Holi, Republic Day\n"
                "• **Prioritize cold storage** in North & South regions (highest volume)\n"
                "• **Use SARIMA** model for monthly production planning\n"
                "• **Promote ice cream & buttermilk** aggressively in summer months")

    # Default
    total_units = int(df['Quantity Sold (Units)'].sum())
    return (f"The SDDIS dataset covers **{total_units:,} dairy unit sales** across 12 products, 5 regions & 19 Indian states from 2021–2023. "
            f"Try asking about: top products, festival impact, seasonal trends, regional performance, revenue, or forecasting models.")
