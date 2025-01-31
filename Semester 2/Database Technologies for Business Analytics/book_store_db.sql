-- Create a database for a library, three tables, two foreign keys
-- Relationships as per ERD: Borrowers can borrow loans / Books can be borrowed by loans
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
