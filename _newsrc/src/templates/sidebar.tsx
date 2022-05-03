import React from "react";
import { graphql } from "gatsby";
import { Header } from "../pages/header";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark as codeStyle } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Drawer } from "@mui/material";

// nightOwl

export default function Sidebar({ modules }) {
  return (
    <Drawer
      variant="temporary"
      anchor={"left"}
      open={true}
      sx={{
        width: 150,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: { width: 150, boxSizing: "border-box" },
      }}
    >
      <div className="sidebar-wrapper">asdasdasad</div>
    </Drawer>
  );
}
