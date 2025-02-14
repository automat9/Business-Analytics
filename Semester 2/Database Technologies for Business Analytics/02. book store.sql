-- Prerequisites
VSCODE

-- Listing entries from a table
SELECT * FROM [BEMM459_MPAW].[practice].[customers]

-- Listing top 5 
SELECT TOP 5 * FROM [BEMM459_MPAW].[practice].[customers]

-- Listing first names and surnames only
SELECT customer_first_name, customer_last_name FROM [BEMM459_MPAW].[practice].[customers];

-- Listing surnames, removing duplicates
SELECT DISTINCT customer_last_name FROM [BEMM459_MPAW].[practice].[customers];

