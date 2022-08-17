// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React, { useState, useEffect } from 'react';


const LikeButton = ({ content_id }) => {
  const [likes, setLikes] = useState(0);
  const [isClicked, setIsClicked] = useState(false);
  const requestOptions = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
        content_id: content_id,
  };

  useEffect(() => {
    fetch('/api/engagement/total_likes', requestOptions)
        .then(response => response.json())
        .then(data => setLikes(data['total_likes']));
  });

  const handleClick = () => {
    let api_uri;
    if (isClicked) {
      setLikes(likes - 1);
      api_uri = '/api/engagement/like_content';
    } else {
      setLikes(likes + 1);
      api_uri = '/api/engagement/unlike_content';
    }
    fetch(api_uri, requestOptions).then(response => response.json());
    setIsClicked(!isClicked);
  };

  return (
    <button className={ `like-button ${isClicked && 'liked'}` } onClick={ handleClick }>
      <span className="likes-counter">{ `Like | ${likes}` }</span>
    </button>
  );
};

export default LikeButton;