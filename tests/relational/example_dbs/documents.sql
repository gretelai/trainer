create table if not exists users (
  id integer primary key,
  name text not null
);
delete from users;

create table if not exists purchases (
  id integer primary key,
  user_id integer not null,
  data text not null,
  --
  foreign key (user_id) references users (id)
);
delete from purchases;

create table if not exists payments (
  id integer primary key,
  purchase_id integer not null,
  amount integer not null,
  --
  foreign key (purchase_id) references purchases (id)
);
delete from payments;

insert into users (id, name) values
  (1, "Andy"),
  (2, "Bob"),
  (3, "Charlie"),
  (4, "David");

insert into purchases (id, user_id, data) values
  (1, 1, '{"item": "pen", "cost": 100, "details": {"color": "red"}, "years": [2023]}'),
  (2, 2, '{"item": "paint", "cost": 100, "details": {"color": "red"}, "years": [2023, 2022]}'),
  (3, 2, '{"item": "ink", "cost": 100, "details": {"color": "red"}, "years": [2020, 2019]}'),
  (4, 3, '{"item": "pen", "cost": 100, "details": {"color": "blue"}, "years": []}'),
  (5, 3, '{"item": "paint", "cost": 100, "details": {"color": "blue"}, "years": [2021]}'),
  (6, 3, '{"item": "ink", "cost": 100, "details": {"color": "blue"}, "years": []}');

insert into payments (id, purchase_id, amount) values
  (1, 1, 42),
  (2, 1, 42),
  (3, 2, 42),
  (4, 2, 42),
  (5, 2, 42),
  (6, 3, 42),
  (7, 4, 42),
  (8, 4, 42),
  (9, 5, 42);
