// from https://stackoverflow.com/questions/72153851/create-a-simple-like-button-component-with-react
import React from "react";
import "../Likes.css";

const LikeButton = (props) => {
  return (
    <button
      className={`likeButton ${props.user_likes ? `liked` : ``}`}
      onClick={props.handleLikes}
    >
      <i className="fa fa-thumbs-up">{props.total_likes}</i>
    </button>
  );
};

export default LikeButton;
