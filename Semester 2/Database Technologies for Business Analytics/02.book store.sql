-- Prerequisites
VSCODE
SQL Server (mssql)
Your Database is : BEMM459_MPAW
Your Username is : MPaw
Your Password is : AmlD130+Dr
The server URL is : mcruebs04.isad.isadroot.ex.ac.uk
  
-- Listing entries from a table
SELECT * FROM [BEMM459_MPAW].[practice].[customers]

-- Listing top 5 
SELECT TOP 5 * FROM [BEMM459_MPAW].[practice].[customers]

-- Listing first names and surnames only
SELECT customer_first_name, customer_last_name FROM [BEMM459_MPAW].[practice].[customers];

-- Listing surnames, removing duplicates
SELECT DISTINCT customer_last_name FROM [BEMM459_MPAW].[practice].[customers];

-- Listing surnames in reverse alphabetical order
SELECT customer_last_name 
FROM [BEMM459_MPAW].[practice].[customers]
ORDER BY customer_last_name DESC;

-- Listing customers living in a specific state
SELECT customer_first_name, customer_last_name
FROM [BEMM459_MPAW].[practice].[customers]
WHERE customer_state = 'SM';
