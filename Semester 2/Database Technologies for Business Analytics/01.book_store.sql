-- Create a database for a library, three tables, two foreign keys
-- Relationships as per ERD: Borrowers can borrow loans / Books can be borrowed by loans
CREATE SCHEMA T3;

CREATE TABLE Borrowers (
    borrower_id INT PRIMARY KEY IDENTITY(1,1),
    name NVARCHAR(100) NOT NULL,
    address NVARCHAR(255),
    phone_number NVARCHAR(20)
);

CREATE TABLE Books (
    book_id INT PRIMARY KEY IDENTITY(1,1),
    title NVARCHAR(255) NOT NULL,
    author NVARCHAR(255) NOT NULL,
    publication_year INT
);

CREATE TABLE T3.Loans (
    loan_id INT PRIMARY KEY IDENTITY(1,1),
    book_id INT NOT NULL FOREIGN KEY REFERENCES Books(book_id),
    borrower_id INT NOT NULL FOREIGN KEY REFERENCES Borrowers(borrower_id),
    borrow_date DATE NOT NULL,
    return_date DATE
);

-- Dummy data
INSERT INTO T3.Books (title, author, publication_year)
VALUES
('The Lord of the Rings', 'J.R.R. Tolkien', 1954),
('Pride and Prejudice', 'Jane Austen', 1813),
('To Kill a Mockingbird', 'Harper Lee', 1960),
('Moby Dick', 'Herman Melville', 1851),
('The Great Gatsby', 'F. Scott Fitzgerald', 1925);

INSERT INTO T3.Borrowers (name, address, phone_number)
VALUES
('John Smith', '123 Main Street', '555-1234'),
('Jane Doe', '456 Elm Street', '555-5678'),
('Michael Brown', '789 Oak Street', '555-9012'),
('Sarah Jones', '012 Maple Avenue', '555-4321');

INSERT INTO T3.Loans (book_id, borrower_id, borrow_date, return_date)
VALUES
(1, 1, '2024-02-09', NULL), -- John Smith borrows "The Lord of the Rings" today
(2, 2, '2024-11-08', '2025-01-15'), -- Jane Doe borrows "Pride and Prejudice"
(3, 3, '2024-11-07', '2025-01-20'), -- Michael Brown borrows "To Kill a Mockingbird"
(4, 4, '2024-12-06', '2025-01-12'), -- Sarah Jones borrows "Moby Dick"
(5, 1, '2024-11-02', '2025-01-04'); -- John Smith borrows "The Great Gatsby" 

-- Tasks
 -- 1.Retrive the borrower information for a specific book
SELECT DISTINCT b.name, b.phone_number
FROM T3.Borrowers b -- Alias 'b' for Borrowers
JOIN T3.Loans l ON b.borrower_id  = l.borrower_id -- Alias 'l' for Loans
JOIN T3.Books bk ON l.book_id = bk.book_id -- Alias 'bk' for Books
WHERE bk.title = 'To Kill a Mockingbird'; -- Referencing 'bk' (Books)

-- 2.Find the most recently borrowed books
SELECT TOP 3 bk.title
FROM T3.Loans l
JOIN T3.Books bk ON l.book_id = bk.book_id
ORDER BY l.borrow_date DESC;
-- Note: SQL Server does not support 'LIMIT', unlike MySQL and other databases, SQL Server instead uses TOP

-- 3.Count the number of unique books borrowed
SELECT COUNT(DISTINCT book_id) AS total_unique_books_borrowed -- custom column name for the output
FROM T3.Loans;

-- 4.List borrowers who borrowed multiple books
SELECT b.name, COUNT(l.book_id) AS books_borrowed
FROM T3.Borrowers b
JOIN T3.Loans l ON b.borrower_id = l.borrower_id
GROUP BY b.name, b.borrower_id
HAVING COUNT(l.book_id) > 1;

-- 5.Filter loans based on date range
SELECT DISTINCT bk.title
FROM T3.Loans l
JOIN T3.Books bk ON l.book_id = bk.book_id
WHERE l.borrow_date BETWEEN '2024-11-01' AND '2024-12-31';



