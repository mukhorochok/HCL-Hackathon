-- ============================================================================
-- ROSSMANN STORE SALES - FIXED SQL SERVER VERSION
-- ============================================================================

USE master;
GO

IF DB_ID('RossmannSales') IS NULL
    CREATE DATABASE RossmannSales;
GO

USE RossmannSales;
GO

-- ============================================================================
-- CREATE SCHEMAS
-- ============================================================================

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name='bronze')
EXEC('CREATE SCHEMA bronze');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name='silver')
EXEC('CREATE SCHEMA silver');
GO

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name='gold')
EXEC('CREATE SCHEMA gold');
GO

-- ============================================================================
-- BRONZE LAYER (RAW LOAD AS VARCHAR TO PREVENT ERRORS)
-- ============================================================================

DROP TABLE IF EXISTS bronze.train;
DROP TABLE IF EXISTS bronze.test;
DROP TABLE IF EXISTS bronze.store;
GO

CREATE TABLE bronze.train(
Store VARCHAR(50),
DayOfWeek VARCHAR(50),
Date VARCHAR(50),
Sales VARCHAR(50),
Customers VARCHAR(50),
OpenFlag VARCHAR(50),
Promo VARCHAR(50),
StateHoliday VARCHAR(50),
SchoolHoliday VARCHAR(50)
);
GO

CREATE TABLE bronze.test(
Id VARCHAR(50),
Store VARCHAR(50),
DayOfWeek VARCHAR(50),
Date VARCHAR(50),
OpenFlag VARCHAR(50),
Promo VARCHAR(50),
StateHoliday VARCHAR(50),
SchoolHoliday VARCHAR(50)
);
GO

CREATE TABLE bronze.store(
Store VARCHAR(50),
StoreType VARCHAR(50),
Assortment VARCHAR(50),
CompetitionDistance VARCHAR(50),
CompetitionOpenSinceMonth VARCHAR(50),
CompetitionOpenSinceYear VARCHAR(50),
Promo2 VARCHAR(50),
Promo2SinceWeek VARCHAR(50),
Promo2SinceYear VARCHAR(50),
PromoInterval VARCHAR(100)
);
GO

-- ============================================================================
-- LOAD DATA (FIXED BULK INSERT)
-- ============================================================================

BULK INSERT bronze.train
FROM 'C:\Users\Anshit\Desktop\HCL\train.csv'
WITH (
FORMAT='CSV',
FIRSTROW=2,
FIELDTERMINATOR=',',
ROWTERMINATOR='0x0a',
TABLOCK
);
GO

BULK INSERT bronze.test
FROM 'C:\Users\Anshit\Desktop\HCL\test.csv'
WITH (
FORMAT='CSV',
FIRSTROW=2,
FIELDTERMINATOR=',',
ROWTERMINATOR='0x0a',
TABLOCK
);
GO

BULK INSERT bronze.store
FROM 'C:\Users\Anshit\Desktop\HCL\store.csv'
WITH (
FORMAT='CSV',
FIRSTROW=2,
FIELDTERMINATOR=',',
ROWTERMINATOR='0x0a',
TABLOCK
);
GO

-- ============================================================================
-- SILVER LAYER
-- ============================================================================

DROP TABLE IF EXISTS silver.sales_cleaned;
DROP TABLE IF EXISTS silver.store_cleaned;
GO

CREATE TABLE silver.sales_cleaned(
sale_id INT IDENTITY(1,1) PRIMARY KEY,
store_id INT,
sale_date DATE,
day_of_week INT,
sales_amount DECIMAL(12,2),
customer_count INT,
is_open BIT,
is_promo BIT,
state_holiday_type VARCHAR(20),
is_school_holiday BIT,
year INT,
month INT,
quarter INT,
is_weekend BIT
);
GO

CREATE TABLE silver.store_cleaned(
store_id INT PRIMARY KEY,
store_type VARCHAR(20),
assortment_type VARCHAR(20),
competition_distance_km DECIMAL(10,2),
competition_open_date DATE,
has_competition BIT,
is_promo2_active BIT,
promo2_start_date DATE
);
GO

-- ============================================================================
-- TRANSFORM STORE DATA
-- ============================================================================

INSERT INTO silver.store_cleaned
SELECT
CAST(Store AS INT),

CASE StoreType
WHEN 'a' THEN 'Type A'
WHEN 'b' THEN 'Type B'
WHEN 'c' THEN 'Type C'
WHEN 'd' THEN 'Type D'
END,

CASE Assortment
WHEN 'a' THEN 'Basic'
WHEN 'b' THEN 'Extra'
WHEN 'c' THEN 'Extended'
END,

TRY_CAST(CompetitionDistance AS DECIMAL(10,2))/1000,

CASE
WHEN TRY_CAST(CompetitionOpenSinceYear AS INT) IS NOT NULL
AND TRY_CAST(CompetitionOpenSinceMonth AS INT) IS NOT NULL
THEN DATEFROMPARTS(
TRY_CAST(CompetitionOpenSinceYear AS INT),
TRY_CAST(CompetitionOpenSinceMonth AS INT),
1)
END,

CASE WHEN CompetitionDistance IS NOT NULL THEN 1 ELSE 0 END,

CASE WHEN Promo2='1' THEN 1 ELSE 0 END,

CASE
WHEN Promo2='1'
AND TRY_CAST(Promo2SinceWeek AS INT) IS NOT NULL
AND TRY_CAST(Promo2SinceYear AS INT) IS NOT NULL
THEN DATEADD(WEEK,
TRY_CAST(Promo2SinceWeek AS INT)-1,
DATEFROMPARTS(TRY_CAST(Promo2SinceYear AS INT),1,1))
END
FROM bronze.store;
GO

-- ============================================================================
-- TRANSFORM SALES DATA
-- ============================================================================

INSERT INTO silver.sales_cleaned
SELECT
CAST(Store AS INT),
TRY_CAST(Date AS DATE),
TRY_CAST(DayOfWeek AS INT),
TRY_CAST(Sales AS DECIMAL(12,2)),
TRY_CAST(Customers AS INT),
CASE WHEN OpenFlag='1' THEN 1 ELSE 0 END,
CASE WHEN Promo='1' THEN 1 ELSE 0 END,

CASE StateHoliday
WHEN '0' THEN 'None'
WHEN 'a' THEN 'Public Holiday'
WHEN 'b' THEN 'Easter Holiday'
WHEN 'c' THEN 'Christmas'
END,

CASE WHEN SchoolHoliday='1' THEN 1 ELSE 0 END,

YEAR(TRY_CAST(Date AS DATE)),
MONTH(TRY_CAST(Date AS DATE)),
DATEPART(QUARTER,TRY_CAST(Date AS DATE)),

CASE
WHEN DATEPART(WEEKDAY,TRY_CAST(Date AS DATE)) IN (1,7)
THEN 1 ELSE 0 END

FROM bronze.train
WHERE TRY_CAST(Date AS DATE) IS NOT NULL;
GO

-- ============================================================================
-- GOLD LAYER
-- ============================================================================

DROP TABLE IF EXISTS gold.daily_store_sales;
DROP TABLE IF EXISTS gold.store_performance_metrics;
GO

CREATE TABLE gold.daily_store_sales(
sales_id INT IDENTITY PRIMARY KEY,
store_id INT,
sale_date DATE,
sales_amount DECIMAL(12,2),
customer_count INT,
avg_transaction_value DECIMAL(10,2),
is_promo BIT,
state_holiday_type VARCHAR(20),
is_weekend BIT,
store_type VARCHAR(20),
assortment_type VARCHAR(20),
has_competition BIT,
year INT,
month INT
);
GO

CREATE TABLE gold.store_performance_metrics(
metric_id INT IDENTITY PRIMARY KEY,
store_id INT,
year INT,
month INT,
total_sales DECIMAL(15,2),
total_customers INT,
avg_daily_sales DECIMAL(12,2),
max_daily_sales DECIMAL(12,2),
promo_days INT,
promo_sales_lift DECIMAL(5,2),
open_days INT
);
GO

-- ============================================================================
-- DAILY SALES
-- ============================================================================

INSERT INTO gold.daily_store_sales
SELECT
sc.store_id,
sc.sale_date,
sc.sales_amount,
sc.customer_count,
CASE WHEN sc.customer_count>0
THEN sc.sales_amount/sc.customer_count ELSE 0 END,
sc.is_promo,
sc.state_holiday_type,
sc.is_weekend,
st.store_type,
st.assortment_type,
st.has_competition,
sc.year,
sc.month
FROM silver.sales_cleaned sc
LEFT JOIN silver.store_cleaned st
ON sc.store_id=st.store_id
WHERE sc.is_open=1;
GO

-- ============================================================================
-- PERFORMANCE METRICS
-- ============================================================================

INSERT INTO gold.store_performance_metrics
SELECT
store_id,
year,
month,
SUM(sales_amount),
SUM(customer_count),
AVG(sales_amount),
MAX(sales_amount),
SUM(CASE WHEN is_promo=1 THEN 1 ELSE 0 END),

ROUND(
(AVG(CASE WHEN is_promo=1 THEN sales_amount END) /
NULLIF(AVG(CASE WHEN is_promo=0 THEN sales_amount END),0)-1)*100
,2),

COUNT(*)

FROM gold.daily_store_sales
GROUP BY store_id,year,month;
GO

-- Top 10 stores by sales
SELECT TOP 10
    store_id,
    SUM(sales_amount) as total_sales,
    AVG(customer_count) as avg_customers
FROM gold.daily_store_sales
GROUP BY store_id
ORDER BY total_sales DESC;
GO
 
-- Monthly sales trend
SELECT 
    [year],
    [month],
    SUM(sales_amount) as monthly_sales,
    AVG(avg_transaction_value) as avg_basket
FROM gold.daily_store_sales
GROUP BY [year], [month]
ORDER BY [year], [month];
GO

-- ============================================================================
-- GENERATE BCP COMMANDS TO EXPORT ALL GOLD TABLES
-- ============================================================================

DECLARE @ServerName VARCHAR(200) = @@SERVERNAME;
DECLARE @DatabaseName VARCHAR(200) = DB_NAME();
DECLARE @OutputFolder VARCHAR(500) = 'C:\Users\Anshit\Desktop\HCL\';

SELECT 
    'bcp "SELECT * FROM ' 
    + QUOTENAME(@DatabaseName) + '.' 
    + QUOTENAME(s.name) + '.' 
    + QUOTENAME(t.name) 
    + '" queryout "' 
    + @OutputFolder + t.name + '.csv" -c -t, -T -S ' + @ServerName
    AS BCP_Command
FROM sys.tables t
JOIN sys.schemas s 
    ON t.schema_id = s.schema_id
WHERE s.name = 'gold';

