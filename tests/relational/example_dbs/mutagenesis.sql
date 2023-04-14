create table if not exists molecule (
  molecule_id text primary key,
  mutagenic text not null
);

create table if not exists atom (
  atom_id integer primary key,
  element text not null,
  charge real not null,
  molecule_id text not null,
  --
  foreign key (molecule_id) references molecule (molecule_id)
);

create table if not exists bond (
  type text not null,
  atom1_id integer not null,
  atom2_id integer not null,
  --
  primary key (atom1_id, atom2_id),
  foreign key (atom1_id) references atom (atom_id),
  foreign key (atom2_id) references atom (atom_id)
);
