// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React, { useState } from 'react';


const LikeButton = ({ content_id, total_likes, user_likes}) => {
  if (total_likes === undefined) {
    total_likes = 0;
  }
  if (user_likes === undefined) {
    user_likes = false;
  }
  const [likes, setLikes] = useState(total_likes);
  const [isClicked, setIsClicked] = useState(user_likes);
  const defaultRequestOptions = {
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
  };
  const postRequestOptions = {...defaultRequestOptions, 'method': 'POST'}

  const handleClick = () => {
    let api_uri;
    if (isClicked) {
      setLikes(likes - 1);
      api_uri = '/api/engagement/unlike_content';
    } else {
      setLikes(likes + 1);
      api_uri = '/api/engagement/like_content';
    }
    fetch(`${api_uri}?content_id=${content_id}`, postRequestOptions).then(response => response.json());
    setIsClicked(!isClicked);
  };

  return (
    <button className={ `like-button ${isClicked && 'liked'}` } onClick={ handleClick }>
      {isClicked ?
          <span className="likes-counter">{`Unlike | ${likes}`}</span>
          : <span className="likes-counter">{`Like | ${likes}`}</span>
      }
    </button>
  );
};

export default LikeButton;