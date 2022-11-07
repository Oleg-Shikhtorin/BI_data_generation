### Generation data for SQL/BI/Data Analysis practice  
Contains sales information, products catalogue, table of users, etc.  
**input:**  
* stores.csv - contains data about stores (manually generated:))  

id | country | city | area | date_start | type | rent_type | rent_price | rent_price_date
---|---|---|---|---|---|---|---|---  

* goods.txt -  list of products  
* goods_probabilities.txt - distribution of items in the table of products  
* locations.txt - user's locations list  
* location_probabilities.txt - it's distribution  
  
**output:**  
* products.csv  

id | sell_price | category | subcategory | price_date | cost_price  
---|---|---|---|---|---  

* users.csv  

id | gender | birth_date | country | city | reg_date  
---|---|---|---|---|---  

* orders.csv  

id | basket_id | user_id | order_date | state | channel | delivery_type | store_id  
---|---|---|---|---|---|---|---  

* baskets.csv  

id | product_id | cnt  
---|---|---  

* events.csv  

id | user_id | dt | event_type | reg_source | page  
---|---|---|---|---|---  

**Some metrics to calculate:**  
Churn rate, VPK (visits per keyword), RV (returning visitors), ER (engagement rate), TSS (time spent on site), PPV (pages per visit), BR (bounce rate), Total Visits, LTV (lifetime value), ROI (return on investment), ARPPU (average revenue per paying user), ARPU (average revenue per user), Margin, AOV (average order value), Total Revenue, OGA (order gap analysis), CRR (customer retention rate), Sales, CR (conversion rate), DAU/MAU/WAU, ACU (average concurrent user)