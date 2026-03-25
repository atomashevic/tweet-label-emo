create table if not exists annotations (
  id            bigint generated always as identity primary key,
  annotator_code text        not null,
  sample_idx    int         not null,
  tweet_pos     int         not null,
  tweet_id      text        not null,
  label         text        not null,
  created_at    timestamptz not null default now(),
  constraint annotations_annotator_tweet_pos_unique unique (annotator_code, tweet_pos)
);

-- Allow anonymous reads and writes (no auth required)
alter table annotations enable row level security;

create policy "anon can insert" on annotations
  for insert to anon with check (true);

create policy "anon can select" on annotations
  for select to anon using (true);

create policy "anon can update" on annotations
  for update to anon using (true) with check (true);
