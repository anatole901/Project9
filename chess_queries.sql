CREATE DATABASE chess;
USE chess;

SELECT * FROM openings;

SELECT count(MyUnknownColumn) FROM chess_cleaned;

SELECT count(c.MyUnknownColumn) as number_of_games, o.`Opening Names`, o.ECO
FROM chess_cleaned c
LEFT JOIN openings o ON c.opening_code = o.ECO
GROUP BY c.opening_code
ORDER BY number_of_games DESC;

SELECT opening_code, average_score_for_white, average_number_of_moves
FROM (SELECT count(MyUnknownColumn) as number_of_rows, opening_code, round(avg(white_points), 3) as average_score_for_white, round(avg(num_moves), 1) as average_number_of_moves
FROM chess_cleaned
GROUP BY opening_code
ORDER BY Average_score_for_white DESC) c
WHERE number_of_rows > 9;

SELECT wgm_username, count(MyUnknownColumn) as number_of_games, round(avg(white_points), 3) as Average_score_with_white
FROM chess_cleaned
WHERE wgm_username = lower(white_username)
GROUP BY wgm_username
ORDER BY Average_score_with_white DESC;

SELECT wgm_username, count(MyUnknownColumn) as number_of_games, 1 - avg(white_points) as Average_score_with_black
FROM chess_cleaned
WHERE wgm_username = lower(black_username)
GROUP BY wgm_username
ORDER BY Average_score_with_black DESC;

SELECT w.wgm_username, IFNULL(w.games_as_white, 0) + IFNULL(b.games_as_black, 0) as number_of_games,
(IFNULL(w.points, 0) + IFNULL(b.points, 0))/(IFNULL(w.games_as_white, 0) + IFNULL(b.games_as_black, 0)) as average_score
FROM (SELECT wgm_username, count(MyUnknownColumn) as games_as_white, sum(white_points) as points
FROM chess_cleaned
WHERE wgm_username = lower(white_username)
GROUP BY wgm_username) w
LEFT JOIN (SELECT wgm_username, count(MyUnknownColumn) as games_as_black, sum(1 - white_points) as points
FROM chess_cleaned
WHERE wgm_username = lower(black_username)
GROUP BY wgm_username) b ON w.wgm_username = b.wgm_username
ORDER BY average_score DESC;

SELECT time_class, avg(if(white_points = 0.5, 1, 0)) as proportion_of_draws
FROM chess_cleaned
GROUP BY time_class;

SELECT round((white_rating - black_rating)/20, 0) * 20 as difference_of_rating, avg(white_points) as Average_score_of_white
FROM chess_cleaned
GROUP BY difference_of_rating
ORDER BY difference_of_rating DESC;