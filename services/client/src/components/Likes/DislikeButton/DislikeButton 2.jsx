// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React from "react";

import "../Likes.css";

const DislikeButton = ({ user_dislikes, handleDislikes }) => {
  return (
    <button
      className={`likeButton ${user_dislikes ? `disliked` : ``}`}
      onClick={handleDislikes}
    >
      <i className="fa fa-thumbs-down" />
    </button>
  );
};

export default DislikeButton;
