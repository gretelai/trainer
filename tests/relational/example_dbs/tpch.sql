create table if not exists supplier (
  s_suppkey integer primary key,
  s_name text not null
);
delete from supplier;

create table if not exists part (
  p_partkey integer primary key,
  p_name text not null
);
delete from part;

create table if not exists partsupp (
  ps_partkey integer not null,
  ps_suppkey integer not null,
  ps_availqty integer not null,
  --
  primary key (ps_partkey, ps_suppkey),
  foreign key (ps_partkey) references part (p_partkey),
  foreign key (ps_suppkey) references supplier (s_suppkey)
);
delete from partsupp;

create table if not exists lineitem (
  l_partkey integer not null,
  l_suppkey integer not null,
  l_quantity integer not null,
  --
  primary key (l_partkey, l_suppkey),
  foreign key (l_partkey, l_suppkey) references partsupp (ps_partkey, ps_suppkey)
);
delete from lineitem;

insert into supplier (s_suppkey, s_name) values
  (1, "SupplierA"),
  (2, "SupplierB"),
  (3, "SupplierC"),
  (4, "SupplierD");

insert into part (p_partkey, p_name) values
  (1, "burlywood plum powder puff mint"),
  (2, "hot spring dodger dim light"),
  (3, "dark slate grey steel misty"),
  (4, "cream turquoise dark thistle light");

insert into partsupp (ps_partkey, ps_suppkey, ps_availqty) values
  (1, 3, 103),
  (1, 2, 102),
  (1, 4, 104),
  (2, 1, 201),
  (2, 2, 202),
  (2, 3, 203),
  (3, 1, 301),
  (3, 3, 303),
  (3, 4, 304),
  (4, 1, 401),
  (4, 4, 404),
  (4, 2, 402);

insert into lineitem (l_partkey, l_suppkey, l_quantity) values
  (1, 3, 13),
  (1, 2, 12),
  (1, 4, 14),
  (2, 1, 21),
  (2, 2, 22),
  (2, 3, 23),
  (3, 1, 31),
  (3, 3, 33),
  (3, 4, 34),
  (4, 1, 41),
  (4, 4, 44),
  (4, 2, 42);
