export const getRefreshTokenIfExists = () => {
  return window.localStorage.getItem("refreshToken");
};

export const setRefreshToken = (token) => {
  window.localStorage.setItem("refreshToken", token);
};

export const removeRefreshToken = () => {
  window.localStorage.removeItem("refreshToken");
};
