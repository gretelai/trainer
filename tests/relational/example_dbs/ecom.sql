create table if not exists users (
  id integer primary key,
  first_name text not null,
  last_name text not null
);

create table if not exists events (
  id integer primary key,
  browser text not null,
  traffic_source text not null,
  user_id text not null,
  --
  foreign key (user_id) references users (id)
);

create table if not exists distribution_center (
  id integer primary key,
  name text not null
);

create table if not exists products (
  id integer primary key,
  name text not null,
  brand text not null,
  distribution_center_id integer not null,
  --
  foreign key (distribution_center_id) references distribution_center (id)
);

create table if not exists inventory_items (
  id integer primary key,
  sold_at text not null,
  cost text not null,
  product_id integer not null,
  product_distribution_center_id integer not null,
  --
  foreign key (product_id) references products (id),
  foreign key (product_distribution_center_id) references distribution_center (id)
);

create table if not exists order_items (
  id integer primary key,
  sale_price text not null,
  status text not null,
  user_id integer not null,
  inventory_item_id integer not null,
  --
  foreign key (user_id) references users (id),
  foreign key (inventory_item_id) references inventory_items (id)
);
