-- sql/queries.sql
-- Part A SQL Analysis Queries for Instamart Customer Intelligence Capstone

-- Q1: WHERE then HAVING
-- Completed orders with order_amount > 500, grouped by customer, having total > 8000
-- Reference: 30 customers
SELECT 
    customer_id, 
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;

-- BREAK_QUERY --

-- Q2: Three-table join
-- Revenue by city and product category (orders, customers, products)
-- Reference: 30 rows; sum = INR 2,998,668.80; largest cell = Bengaluru / Electronics (INR 233,909.45)
SELECT 
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- BREAK_QUERY --

-- Q3: Scalar subquery
-- Products priced above the average catalogue price
-- Reference: 11 products
SELECT 
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- BREAK_QUERY --

-- Q4: LEFT JOIN
-- Customers who have no completed order
-- Reference: 25 customers
SELECT 
    c.customer_id,
    c.city,
    c.signup_date
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- BREAK_QUERY --

-- Q5: Common Table Expression (CTE)
-- Spend per customer, average of totals, customers with spend above average
-- Reference: 550 customers; average INR 2,552.06
WITH customer_spend AS (
    SELECT 
        customer_id,
        ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total_spend
    FROM customer_spend
)
SELECT 
    cs.customer_id,
    cs.total_spend,
    ROUND((SELECT avg_total_spend FROM avg_spend), 2) AS avg_benchmark
FROM customer_spend cs
WHERE cs.total_spend > (SELECT avg_total_spend FROM avg_spend)
ORDER BY cs.total_spend DESC;

-- BREAK_QUERY --

-- Q6: Ranking window functions
-- For each customer with completed spend: RANK(), DENSE_RANK(), and ROW_NUMBER()
-- Reference: top per city (Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60;
-- Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10); exactly 8 rows RANK != ROW_NUMBER
WITH customer_city_spend AS (
    SELECT 
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT 
    city,
    customer_id,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_dense_rank,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_row_number
FROM customer_city_spend
ORDER BY city, spend_rank;

-- BREAK_QUERY --

-- Q7: Running total
-- Daily completed revenue with cumulative revenue
-- Reference: 183 daily rows; final cumulative value = INR 2,998,668.80
WITH daily_revenue AS (
    SELECT 
        SUBSTR(order_ts, 1, 10) AS day,
        ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT 
    day,
    revenue,
    ROUND(SUM(revenue) OVER (ORDER BY day), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
