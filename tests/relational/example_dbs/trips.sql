create table if not exists vehicle_types (
  id integer primary key,
  name text not null
);
delete from vehicle_types;

create table if not exists trips (
  id integer primary key,
  purpose text not null,
  vehicle_type_id integer not null,
  --
  foreign key (vehicle_type_id) references vehicle_types (id)
);
delete from vehicle_types;

insert into vehicle_types (name) values
  ("car"), ("train"), ("bike"), ("plane");

-- trip data inserted via python fixture
