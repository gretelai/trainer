create table if not exists beneficiary (
  id integer primary key,
  name text not null
);

create table if not exists insurance_policies (
  id integer primary key,
  primary_beneficiary integer not null,
  secondary_beneficiary integer not null,
  --
  foreign key (primary_beneficiary) references beneficiary (id),
  foreign key (secondary_beneficiary) references beneficiary (id)
);

insert into beneficiary (name) values
  ("John Doe"),
  ("Jane Smith"),
  ("Michael Johnson"),
  ("Emily Brown"),
  ("William Wilson");

insert into insurance_policies (primary_beneficiary, secondary_beneficiary) values
  (1, 2),
  (2, 3),
  (3, 4),
  (4, 5),
  (5, 1),
  (1, 3),
  (2, 4),
  (3, 5),
  (4, 1),
  (5, 2);
