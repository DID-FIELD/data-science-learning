PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS LoanCustomer;
DROP TABLE IF EXISTS LoanPayment;
DROP TABLE IF EXISTS Loan;
DROP TABLE IF EXISTS Account;
DROP TABLE IF EXISTS Company;
DROP TABLE IF EXISTS Individual;
DROP TABLE IF EXISTS Customer;
DROP TABLE IF EXISTS Branch;

CREATE TABLE Customer (
    cid VARCHAR(10) PRIMARY KEY,
    cname VARCHAR(50) NOT NULL
);

CREATE TABLE Company (
    cid VARCHAR(10) PRIMARY KEY,
    street VARCHAR(100) NOT NULL,
    city VARCHAR(50) NOT NULL,
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);

CREATE TABLE Individual (
    cid VARCHAR(10) PRIMARY KEY,
    gender VARCHAR(10) NOT NULL,
    age INTEGER NOT NULL,
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);

CREATE TABLE Account (
    aid VARCHAR(10) PRIMARY KEY,
    overdraft_limit DECIMAL(10, 2) NOT NULL,
    start_date DATE NOT NULL,
    pin VARCHAR(20) NOT NULL,
    cid VARCHAR(10) NOT NULL,
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);

CREATE TABLE Branch (
    branch_number VARCHAR(10) PRIMARY KEY,
    city VARCHAR(50) NOT NULL,
    street VARCHAR(100) NOT NULL
);

CREATE TABLE Loan (
    loan_number VARCHAR(10) PRIMARY KEY,
    loan_type VARCHAR(20) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    branch_number VARCHAR(10) NOT NULL,
    FOREIGN KEY (branch_number) REFERENCES Branch(branch_number) ON DELETE CASCADE
);

CREATE TABLE LoanPayment (
    loan_number VARCHAR(10) NOT NULL,
    payment_number INTEGER NOT NULL,
    date DATE NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    PRIMARY KEY (loan_number, payment_number),
    FOREIGN KEY (loan_number) REFERENCES Loan(loan_number) ON DELETE CASCADE
);

CREATE TABLE LoanCustomer (
    loan_number VARCHAR(10) NOT NULL,
    cid VARCHAR(10) NOT NULL,
    PRIMARY KEY (loan_number, cid),
    FOREIGN KEY (loan_number) REFERENCES Loan(loan_number) ON DELETE CASCADE,
    FOREIGN KEY (cid) REFERENCES Customer(cid) ON DELETE CASCADE
);


CREATE TRIGGER check_individual_age_positive_insert
BEFORE INSERT ON Individual
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN NEW.age <= 0
        THEN RAISE(ABORT, 'Error: Age must be a positive integer')
    END;
END;

CREATE TRIGGER check_individual_age_positive_update
BEFORE UPDATE ON Individual
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN NEW.age <= 0
        THEN RAISE(ABORT, 'Error: Age must be a positive integer')
    END;
END;

CREATE TRIGGER check_company_insert
BEFORE INSERT ON Company
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN EXISTS (SELECT 1 FROM Individual WHERE cid = NEW.cid)
        THEN RAISE(ABORT, 'Error: Customer already exists as an Individual')
    END;
END;

CREATE TRIGGER check_company_update
BEFORE UPDATE ON Company
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN EXISTS (SELECT 1 FROM Individual WHERE cid = NEW.cid)
        THEN RAISE(ABORT, 'Error: Customer already exists as an Individual')
    END;
END;

CREATE TRIGGER check_individual_insert
BEFORE INSERT ON Individual
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN EXISTS (SELECT 1 FROM Company WHERE cid = NEW.cid)
        THEN RAISE(ABORT, 'Error: Customer already exists as a Company')
    END;
END;

CREATE TRIGGER check_individual_update
BEFORE UPDATE ON Individual
FOR EACH ROW
BEGIN
    SELECT CASE
        WHEN EXISTS (SELECT 1 FROM Company WHERE cid = NEW.cid)
        THEN RAISE(ABORT, 'Error: Customer already exists as a Company')
    END;
END;

