create table if not exists artists (
  id text primary key,
  name text not null
);
delete from artists;

create table if not exists paintings (
  id text primary key,
  name text not null,
  artist_id text not null,
  --
  foreign key (artist_id) references artists (id)
);
delete from paintings;

insert into artists (id, name) values
  ("A001", "Wassily Kandinsky"),
  ("A002", "Pablo Picasso"),
  ("A003", "Vincent van Gogh"),
  ("A004", "Leonardo da Vinci");

insert into paintings (id, artist_id, name) values
  ("P001", "A004", "Mona Lisa"),
  ("P002", "A004", "The Last Supper"),
  ("P004", "A002", "Guernica"),
  ("P005", "A002", "The Old Guitarist"),
  ("P006", "A003", "Starry Night"),
  ("P007", "A003", "Bedroom in Arles"),
  ("P008", "A003", "Irises");
