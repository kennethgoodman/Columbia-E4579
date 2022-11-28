// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React from "react";

import "../Likes.css";

const LikeButton = ({ user_likes, handleLikes }) => {
  return (
    <div>
      <button
        className={`likeButton ${user_likes ? `liked` : ``}`}
        onClick={handleLikes}
      >
        <i className="fa fa-thumbs-up" />
      </button>
    </div>
  );
};

export default LikeButton;
