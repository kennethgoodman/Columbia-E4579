import React from "react";
import PropTypes from "prop-types";
import { Formik } from "formik";
import * as Yup from "yup";

import "./form.css";

const AddUser = (props) => (
  <Formik
    initialValues={{
      username: "",
      password: "",
    }}
    onSubmit={(values, { setSubmitting, resetForm }) => {
      props.addUser(values);
      resetForm();
      setSubmitting(false);
    }}
    validationSchema={Yup.object().shape({
      username: Yup.string()
        .required("Username is required."),
      password: Yup.string()
        .required("Password is required.")
    })}
  >
    {(props) => {
      const {
        values,
        touched,
        errors,
        isSubmitting,
        handleChange,
        handleBlur,
        handleSubmit,
      } = props;
      return (
        <form onSubmit={handleSubmit}>
          <div className="field">
            <label className="label" htmlFor="input-username">
              Username
            </label>
            <input
              name="username"
              id="input-username"
              className={
                errors.username && touched.username ? "input error" : "input"
              }
              type="text"
              placeholder="Enter a username"
              value={values.username}
              onChange={handleChange}
              onBlur={handleBlur}
            />
            {errors.username && touched.username && (
              <div className="input-feedback">{errors.username}</div>
            )}
          </div>
          <div className="field">
            <label className="label" htmlFor="input-password">
              Password
            </label>
            <input
              name="password"
              id="input-password"
              className={
                errors.password && touched.password ? "input error" : "input"
              }
              type="password"
              placeholder="Enter a password"
              value={values.password}
              onChange={handleChange}
              onBlur={handleBlur}
            />
            {errors.password && touched.password && (
              <div className="input-feedback">{errors.password}</div>
            )}
          </div>
          <input
            type="submit"
            className="button is-primary"
            value="Submit"
            disabled={isSubmitting}
          />
        </form>
      );
    }}
  </Formik>
);

AddUser.propTypes = {
  addUser: PropTypes.func.isRequired,
};

export default AddUser;
