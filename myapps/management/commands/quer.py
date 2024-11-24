from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./nsql")
model = AutoModelForCausalLM.from_pretrained("./nsql")

# User query
user_query = "artist who wrote the song c.o.d"

# Define schema and corresponding keywords
schema_keywords = {
    "Genre.Name": ["Rock", "Jazz", "Metal", "Alternative & Punk", "Rock And Roll", "Blues", "Latin", "Reggae", "Pop", "Soundtrack", "Bossa Nova", "Easy Listening", "Heavy Metal", "R&B/Soul", "Electronica/Dance", "World", "Hip Hop/Rap", "Science Fiction", "TV Shows", "Sci Fi & Fantasy", "Drama", "Comedy", "Alternative", "Classical", "Opera"],
    "Playlist.Name": ["Music", "Movies", "TV Shows", "Audiobooks", "90's Music", "Audiobooks", "Music Videos", "Brazilian Music", "Classical", "Classical 101 - Deep Cuts", "Classical 101 - Next Steps", "Classical 101 - The Basics", "Grunge", "Heavy Metal Classic", "On-The-Go 1"]
}

# Preprocess the user query to map keywords to columns
def preprocess_query(query, schema_keywords):
    query_tokens = query.split()
    mapped_tokens = []
    
    for token in query_tokens:
        mapped = False
        for column, keywords in schema_keywords.items():
            if token.lower() in (keyword.lower() for keyword in keywords):
                mapped_tokens.append(f"{column}={token}")
                mapped = True
                break
        if not mapped:
            mapped_tokens.append(token)
    
    return " ".join(mapped_tokens)

# Preprocess the user query
mapped_query = preprocess_query(user_query, schema_keywords)

# Construct the input text with the SQL query
text = f"""
Schema:
Album:
    AlbumId: PRIMARY KEY INTEGER,
    Title: NVARCHAR(160),
    ArtistId: FOREIGN KEY INTEGER,
    FOREIGN KEY (ArtistId) REFERENCES Artist(ArtistId)
Artist:
    ArtistId: PRIMARY KEY INTEGER,
    Name: NVARCHAR(120)
Genre:
    GenreId: PRIMARY KEY INTEGER,
    Name: "ENUM('Rock','Jazz','Metal','Alternative & Punk','Rock And Roll','Blues','Latin','Reggae','Pop','Soundtrack','Bossa Nova','Easy Listening','Heavy Metal','R&B/Soul','Electronica/Dance','World','Hip Hop/Rap','Science Fiction','TV Shows','Sci Fi & Fantasy','Drama','Comedy','Alternative','Classical','Opera')"
Playlist:
    PlaylistId: PRIMARY KEY INTEGER,
    Name: "ENUM('Music','Movies','TV Shows','Audiobooks','90's Music','Audiobooks','Music Videos','Brazilian Music','Classical','Classical 101 - Deep Cuts','Classical 101 - Next Steps','Classical 101 - The Basics','Grunge','Heavy Metal Classic','On-The-Go 1')" ,
    FOREIGN KEY (PlaylistId) REFERENCES PlaylistTrack(PlaylistId)
PlaylistTrack:
    PlaylistId: FOREIGN KEY INTEGER,
    TrackId: FOREIGN KEY INTEGER,
    FOREIGN KEY (PlaylistId) REFERENCES Playlist(PlaylistId),
    FOREIGN KEY (TrackId) REFERENCES Track(TrackId)
Track:
    TrackId: PRIMARY KEY INTEGER,
    Name: NVARCHAR(200),
    AlbumId: FOREIGN KEY INTEGER,
    MediaTypeId: FOREIGN KEY INTEGER,
    GenreId: FOREIGN KEY INTEGER,
    Composer: NVARCHAR(220),
    Milliseconds: INTEGER,
    Bytes: INTEGER,
    UnitPrice: NUMERIC(10,2),
    FOREIGN KEY (AlbumId) REFERENCES Album(AlbumId),
    FOREIGN KEY (GenreId) REFERENCES Genre(GenreId)

-- Using valid SQLite, answer the following questions for the tables provided above.

-- Examples of SQL Queries:

-- Example 1: Get all/every tracks by artist
SELECT A.Name AS ArtistName, T.Name AS TrackName FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId JOIN Track T ON Al.AlbumId = T.AlbumId WHERE A.Name = 'artist';

-- Example 2: Get all/every albums by artist
SELECT A.Name AS ArtistName, Al.Title AS AlbumTitle FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId WHERE A.Name = 'artist';

-- Example 3: Get all pop/rock/jazz tracks/songs
SELECT G.Name AS GenreName, T.Name AS TrackName FROM Genre G JOIN Track T ON G.GenreId = T.GenreId WHERE G.Name = 'pop/rock/jazz';

-- Example 4: Get all tracks/songs in classical/brazilian playlist
SELECT P.Name AS PlaylistName, T.Name AS TrackName FROM Playlist P JOIN PlaylistTrack PT ON P.PlaylistId = PT.PlaylistId JOIN Track T ON PT.TrackId = T.TrackId WHERE P.Name = 'classical/brazilian';

-- Example 5: Get all playlists that contain a track/song named evil walks
SELECT P.Name AS PlaylistName, T.Name AS TrackName FROM Playlist P JOIN PlaylistTrack PT ON P.PlaylistId = PT.PlaylistId JOIN Track T ON PT.TrackId = T.TrackId WHERE T.Name = 'evil walks';

-- Example 6: Get all tracks/songs from albums of artist named audioslaves
SELECT A.Name AS ArtistName, Al.Title AS AlbumTitle, T.Name AS TrackName FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId JOIN Track T ON Al.AlbumId = T.AlbumId WHERE A.Name = 'audioslaves';

-- Example 7: Get total duration of tracks/songs in the album facelift
SELECT Al.Title AS AlbumTitle, SUM(T.Milliseconds) AS TotalDuration FROM Album Al JOIN Track T ON Al.AlbumId = T.AlbumId WHERE Al.Title = 'facelift' GROUP BY Al.Title;

-- Example 8: Get total number of tracks/songs in each genre
SELECT G.Name AS GenreName, COUNT(T.TrackId) AS NumberOfTracks FROM Genre G JOIN Track T ON G.GenreId = T.GenreId GROUP BY G.Name;

-- Example 9: Get the average track/song length by genre
SELECT G.Name AS GenreName, AVG(T.Milliseconds) AS AverageTrackLength FROM Genre G JOIN Track T ON G.GenreId = T.GenreId GROUP BY G.Name;

-- Example 10: Get all artists and the number of albums they have released
SELECT A.Name AS ArtistName, COUNT(Al.AlbumId) AS NumberOfAlbums FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId GROUP BY A.Name;

-- Example 11: Get all tracks/songs and their album and artist names
SELECT T.Name AS TrackName, Al.Title AS AlbumTitle, A.Name AS ArtistName FROM Track T JOIN Album Al ON T.AlbumId = Al.AlbumId JOIN Artist A ON Al.ArtistId = A.ArtistId;

-- Example 12: Get the total price of all tracks in classical/brazilian playlist
SELECT P.Name AS PlaylistName, SUM(T.UnitPrice) AS TotalPrice FROM Playlist P JOIN PlaylistTrack PT ON P.PlaylistId = PT.PlaylistId JOIN Track T ON PT.TrackId = T.TrackId WHERE P.Name = 'classical/brazilian' GROUP BY P.Name;

-- Example 13: give all the songs/tracks released by ac/dc
SELECT T.Name AS TrackName FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId JOIN Track T ON Al.AlbumId = T.AlbumId WHERE A.Name = 'AC/DC';

-- Example 14: all albums released by led zeppelin
SELECT Al.Title AS AlbumTitle FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId WHERE A.Name = 'Led Zeppelin';

-- Example 15: who released the album out of time
SELECT A.Name AS ArtistName FROM Artist A JOIN Album Al ON A.ArtistId = Al.ArtistId WHERE Al.Title = 'Out of Time';

-- Example 16: composer of the track/songs what it takes
SELECT T.Composer FROM Track T WHERE T.Name = 'What It Takes';

-- Example 17: length of the track/songs Breaking the rules
SELECT T.Milliseconds AS TrackLength FROM Track T WHERE T.Name = 'Breaking The Rules';

-- Example 18: Get all pop tracks
SELECT G.Name AS GenreName, T.Name AS TrackName FROM Genre G JOIN Track T ON G.GenreId = T.GenreId WHERE G.Name = 'pop';

-- Your Query: {mapped_query}
"""

input_ids = tokenizer(text, return_tensors="pt").input_ids

generated_ids = model.generate(input_ids, max_length=2048)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
