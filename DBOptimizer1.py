#!/usr/bin/env python
# coding: utf-8

# In[131]:


pip install cx_Oracle langchain openai psycopg2 langchain-experimental sql-metadata


# In[121]:


import os, re
import psycopg2
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from sqlalchemy import inspect, text
from sql_metadata import Parser


# In[91]:


# ANSI color codes
BOLD = "\033[1m"
BLUE = "\033[94m"
GREEN = "\033[92m"
RESET = "\033[0m"


# In[92]:


# Initialize connection to PostgreSQL
#user name postgres password welcome@123 (@ is provided as %40)
db = SQLDatabase.from_uri(
    "postgresql+psycopg2://postgres:welcome%40123@localhost:5432/learn_sql",
    include_tables=['order_items', 'orders', 'products'],  # Focus on relevant tables
    sample_rows_in_table_info=2,  # Include sample data for context
    view_support=True  # Important for PostgreSQL views
)


# In[93]:


def get_full_schema_with_sqlalchemy(db: SQLDatabase, tables: list):    
    engine = db._engine
    inspector = inspect(engine)
    result = {}
    for table in tables:        
        cols = inspector.get_columns(table)
        pk = inspector.get_pk_constraint(table).get('constrained_columns', [])
        idxs = inspector.get_indexes(table)        
        result[table] = {
            "columns": cols,
            "primary_keys": pk,
            "indexes": idxs
        }
    return result


# In[129]:


def get_list_of_tables_views_from_query(query):
    parser = Parser(query)
    tables = parser.tables    
    return tables
    


# In[130]:


test_query= """
SELECT o.order_id, p.name AS product_name, oi.quantity, oi.unit_price, oi.subtotal FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE o.order_id = 2 
"""
tables_views = get_list_of_tables_views_from_query(test_query.replace("\n","").strip())
print(tables_views)


# In[118]:


def get_query_plan(db: SQLDatabase, query: str):
    engine = db._engine    
    query_plan = ""
    with engine.connect() as conn:
        explain_sql = f"EXPLAIN ANALYZE {query}"
        result_plan = conn.execute(text(explain_sql))
        plan_lines = [row[0] for row in result_plan.fetchall()]
        query_plan = "\n".join(plan_lines)
        
    return query_plan


# In[132]:


test_query= """
SELECT o.order_id, p.name AS product_name, oi.quantity, oi.unit_price, oi.subtotal FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE o.order_id = 2 
"""
query_plan = get_query_plan(db, test_query.replace("\n","").strip())
print(query_plan)
#You are an expert PostgreSQL performance tuner. Analyze the following SQL query and its query plan. 
#Tell me if it uses indexes efficiently, and if not, recommend changes to the query or database (such as new indexes or query rewrites).


# In[94]:


table_info = []
table_info = get_full_schema_with_sqlalchemy(db,['order_items', 'orders', 'products'])
for table_name, info in table_info.items():
    print(f"\n{BOLD}{BLUE}Table Name: {table_name} Columns {RESET}")    
    for col in info['columns']:
        print(col)
        #break #in single row we get alll the information 
    print(f"\n{BOLD}{BLUE}Table Name: {table_name} Primary Key {RESET}")    
    for pkey in info['primary_keys']:
        print(pkey)
        #break #in single row we get alll the information 
    print(f"\n{BOLD}{BLUE}Table Name: {table_name} Index {RESET}")    
    for idx in info['indexes']:
        print(idx)
        #break #in single row we get alll the information 
        


# In[134]:


"""
1. Collect Inputs
To analyze data skew, pass some or all of the following into the LLM:

The SQL query.

The EXPLAIN (ANALYZE, BUFFERS) plan.

Table row counts.

Column value distributions (e.g., SELECT column, COUNT(*) FROM table GROUP BY column ORDER BY COUNT(*) DESC LIMIT 5;)

Index definitions (optional).

You are a database optimization expert.

Analyze the following query and its plan to identify any data skew issues that may affect performance. Also, consider the column value distributions provided.

You can define a prompt template and build a chain that inputs:

query

query_plan

column_distribution

row_counts

Then run the chain to generate skew analysis.

                                                       
"""


# In[ ]:


# Initialize LLM - gpt-4 works better for SQL tasks
os.environ["OPENAI_API_KEY"] = "Your Key"
llm = ChatOpenAI(model="gpt-4o-mini")  # or gpt-4, etc.


# In[ ]:


class RawSQLDatabaseChain(SQLDatabaseChain):
    def _execute(self, query, *args, **kwargs):
        #to resolve this error syntax error at or near "```" LINE 1
        match = re.match(r"```sql\s*(.*?)\s*```", query, re.DOTALL | re.IGNORECASE)
        if match:
            query = match.group(1)
        # Remove ```sql and ``` if they exist
        if query.startswith("```sql"):
            query = query[6:]  # Remove ```sql
        if query.endswith("```"):
            query = query[:-3]  # Remove ```        
        return super()._execute(query.strip(), *args, **kwargs)

db_chain = RawSQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,
    return_intermediate_steps=True,
    use_query_checker=False  # Validate queries before execution
)


# In[ ]:


#prompt template
# PostgreSQL-specific optimization prompt
pg_optimization_prompt = PromptTemplate(
    input_variables=["query", "table_info"],
    template="""
    You are a PostgreSQL database expert. Optimize this SQL query for maximum performance.
    
    Table Information:
    {table_info}
    
    Original Query:
    {query}
    
    Provide:
    1. Optimized Query with PostgreSQL-specific improvements
    2. Detailed explanation of changes made
    3. Recommended indexes with CREATE INDEX statements
    4. ANALYZE recommendations
    5. PostgreSQL-specific optimizations (CTEs, window functions, etc.)
    6. EXPLAIN plan interpretation
    
    Respond in this format:
    ### Optimized Query ###
    <optimized SQL here>
    
    ### Optimization Explanation ###
    <explanation here>
    
    ### Recommended Indexes ###
    <index creation statements>
    
    ### ANALYZE Recommendations ###
    <analyze commands>
    
    ### PostgreSQL Features ###
    <suggested features>
    
    ### EXPLAIN Plan Insights ###
    <plan interpretation>
    """
)


# In[ ]:


#construct prompt
table_info = """ 
Table: products
Columns: product_id SERIAL PRIMARY KEY,name VARCHAR(100) NOT NULL,description TEXT,price DECIMAL(10,2) NOT NULL,stock_quantity INTEGER NOT NULL DEFAULT 0,category_id INTEGER REFERENCES categories(category_id),sku VARCHAR(50) UNIQUE,image_url VARCHAR(255),created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP

Table: orders 
Columns: order_id SERIAL PRIMARY KEY,user_id INTEGER REFERENCES users(user_id),order_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,total_amount DECIMAL(10,2) NOT NULL,status VARCHAR(20) NOT NULL DEFAULT 'pending',shipping_address_id INTEGER REFERENCES addresses(address_id),payment_method VARCHAR(50),created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP

Table: order_items 
Column: order_item_id SERIAL PRIMARY KEY,order_id INTEGER REFERENCES orders(order_id),product_id INTEGER REFERENCES products(product_id),quantity INTEGER NOT NULL,unit_price DECIMAL(10,2) NOT NULL,subtotal DECIMAL(10,2) GENERATED ALWAYS AS (quantity * unit_price) STORED
"""
query= """
SELECT o.order_id, p.name AS product_name, oi.quantity, oi.unit_price, oi.subtotal FROM orders o JOIN order_items oi ON o.order_id = oi.order_id JOIN products p ON oi.product_id = p.product_id WHERE o.order_id = 2 
"""


# In[ ]:


print("Print query with Invisible character")
print(f"QUERY:\n{repr(query)}")
query=query.replace("```sql", "").replace("```", "").replace("`", "").replace("\n","").strip()
print(f"QUERY:\n{repr(query)}")


# In[ ]:


def debug_execute(query):
    print("Debug (Python):", repr(query))  # Check what you THINK is being sent
    conn = psycopg2.connect("postgresql://postgres:welcome%40123@localhost:5432/learn_sql")
    cursor = conn.cursor()
    try:
        cursor.execute(query)  # The line where the error occurs
    except Exception as e:
        print("PostgreSQL received:", repr(cursor.query))  # What was ACTUALLY sent
        raise

debug_execute(query)


# In[ ]:


print(f"QUERY:{repr(query)}")
result = db_chain(query)


# In[ ]:





# In[ ]:


from langchain.chains import LLMChain
chain = LLMChain(
        llm=llm,
        prompt=pg_optimization_prompt
    )
response = chain.run({
    "query": query,
    "table_info": table_info
})
print("Optimized Query and Explanation:\n")
print(response)


# In[ ]:


# Define a dictionary with expected section names and default to None
llm_output = {
    "optimized_query": None,
    "optimization_explanation": None,
    "recommended_indexes": None,
    "analyze_recommendations": None,
    "postgresql_features": None,
    "explain_plan_insights": None
}

# Define regex patterns for each section
patterns = {
    "optimized_query": r"### Optimized Query ###\s*```sql\n(.*?)\n```",
    "optimization_explanation": r"### Optimization Explanation ###\s*(.*?)(?=\n###|\Z)",
    "recommended_indexes": r"### Recommended Indexes ###\s*```sql\n(.*?)\n```",
    "analyze_recommendations": r"### ANALYZE Recommendations ###\s*```sql\n(.*?)\n```",
    "postgresql_features": r"### PostgreSQL Features ###\s*(.*?)(?=\n###|\Z)",
    "explain_plan_insights": r"### EXPLAIN Plan Insights ###\s*(.*?)(?=\n###|\Z)"
}

# Apply regex safely for each section
for key, pattern in patterns.items():
    match = re.search(pattern, response, re.DOTALL)
    if match:
        llm_output[key] = match.group(1).strip() if match else None

#print("Optimized Query:", llm_output["optimized_query"] or "Not available")
#print("Optimization Explanation:", llm_output["optimization_explanation"] or "Not available")
#print("Index Recommendation:", llm_output["recommended_indexes"] or "Not available")
#print("Analyse Recommendation:", llm_output["analyze_recommendations"] or "Not available")
#print("Explain Plan:", llm_output["explain_plan_insights"] or "Not available")
#print("Suggested Features:", llm_output["postgresql_features"] or "Not available")

print("\033[1mUSER QUERY:\033[0m")
print(query)

# Display all extracted sections in a nice format
print("=" * 120)
print(f"\033[1m{'Optimization Recommendations'.center(120)}\033[0m")
print("=" * 120)

for key, value in llm_output.items():
    print(f"\n\033[1m{key.upper()}:\033[0m\n")
    print(value if value else " Not available.")

print("\n" + "=" * 120)


# In[ ]:


from langchain.prompts import PromptTemplate

pg_optimization_prompt = PromptTemplate(
    input_variables=["query", "table_info", "existing_indexes"],
    template="""
You are a PostgreSQL performance expert. Analyze and optimize the given SQL query.
Only suggest new indexes if they do not already exist in the list of existing indexes.

### SQL Query ###
{query}

### Table Information ###
{table_info}

### Existing Indexes ###
{existing_indexes}

### Optimization Goals ###
- Improve performance
- Use existing indexes where possible
- Avoid recommending indexes that already exist

### Respond in the following format:
1. Optimized SQL Query
2. Explanation of Optimizations
3. Only New Recommended Indexes (if any)
"""
)


query = """
SELECT o.order_id, c.name, p.name, oi.quantity
FROM orders o
JOIN consumers c ON o.consumer_id = c.consumer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '7 days';
"""

table_info = """
orders(order_id PK, consumer_id FK, order_date DATE)
consumers(consumer_id PK, name TEXT)
order_items(order_id FK, product_id FK, quantity INT)
products(product_id PK, name TEXT)
"""

existing_indexes = """
orders(order_id), orders(consumer_id), products(product_id), consumers(consumer_id)
"""

final_prompt = pg_optimization_prompt.format(
    query=query,
    table_info=table_info,
    existing_indexes=existing_indexes
)

response = llm(final_prompt)
print(response)




# In[ ]:




