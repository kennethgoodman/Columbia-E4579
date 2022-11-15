SELECT
	user_id,
	user.username,
	ROUND(total_ms / 1000 / 60) as minutes,
	num_img,
	ROUND(total_ms / 1000 / num_img, 2) as avg_sec_per_photo,
	ROUND(total_likes) as total_likes,
	ROUND(total_dislikes) as total_dislikes,
	last_engagement
FROM (
	SELECT
		user_id,
		SUM(
			-- can't spend more than 1.5 minutes on a photo
			LEAST(engagement_value, 90 * 1000)
		) AS total_ms,
		COUNT(1) AS num_img,
		MAX(created_date) as last_engagement
	FROM engagement
	WHERE
		engagement_type = 'MillisecondsEngagedWith'
	GROUP BY user_id
) AS t1
INNER JOIN (
	SELECT
		user_id,
		SUM(engagement_value) AS like_and_dislike,
		COUNT(1) / 2 + SUM(engagement_value) / 2 AS total_likes,
		COUNT(1) / 2 - SUM(engagement_value) / 2 AS total_dislikes
	FROM engagement
	WHERE
		engagement_type = 'Like'
	GROUP BY user_id
) AS t2
	USING (user_id)
INNER JOIN user
	ON user_id = user.id
ORDER BY total_ms DESC;
