create table if not exists humans (
  id integer primary key,
  name text not null,
  city text not null
);
delete from humans;

create table if not exists pets (
  id integer primary key,
  name text not null,
  age integer not null,
  human_id integer not null,
  foreign key (human_id) references humans (id)
);
delete from pets;

insert into humans (name, city) values
  ("John", "Liverpool"),
  ("Paul", "Liverpool"),
  ("George", "Liverpool"),
  ("Ringo", "Liverpool"),
  ("Billy", "Houston");

insert into pets (human_id, name, age) values
  (1, "Lennon", 6),
  (2, "McCartney", 14),
  (3, "Harrison", 8),
  (4, "Starr", 7),
  (5, "Preston", 2);
