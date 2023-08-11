
CREATE database login; 
USE login;

CREATE TABLE details(
		username varchar (20),
        password varchar (20));
        
INSERT into details values (
	'KARTHICK_NG','1234'),
    ('SUNDHAR_SIR','1234');
SELECT *from details
