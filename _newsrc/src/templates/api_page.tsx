import React from "react";
import { graphql } from "gatsby";
import { Header } from "../pages/header";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark as codeStyle } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Box,
  CssBaseline,
  Drawer,
  AppBar,
  Toolbar,
  Typography,
  Collapse,
  ListItem,
  ListItemButton,
  ListItemText,
  List,
} from "@mui/material";

import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import Sidebar from "./sidebar";

// nightOwl

export default function ApiPage({ pageContext }) {
  return (
    <div className="api-wrapper">
      <div className="api-title">
        {pageContext.qualified_name.replace("deeptrack", "dt")}
      </div>

      <div className="api-class-vis">
        <SyntaxHighlighter language="python" style={codeStyle}>
          {getSignature(
            pageContext.qualified_name.replace("deeptrack", "dt"),
            pageContext.signature
          )}
        </SyntaxHighlighter>
      </div>

      <div className="api-body">
        <RenderNodes nodes={pageContext.sections.body} />
      </div>

      {pageContext.sections.Parameters ? (
        <div className="api-arguments">
          <RenderNodes nodes={pageContext.sections.Parameters} />
        </div>
      ) : null}

      {pageContext.sections.Attributes ? (
        <div className="api-attributes">
          <RenderNodes nodes={pageContext.sections.Attributes} />
        </div>
      ) : null}
    </div>
  );
}

function RenderNodes({ nodes }) {
  return nodes.map((node, i) => {
    if (node.type === "Text") {
      return (
        <span className={node.type} key={i}>
          {node.content}
        </span>
      );
    }

    switch (node.type) {
      case "section":
        return (
          <div className="api-section" key={i}>
            <RenderNodes nodes={node.children} />
          </div>
        );
      case "title":
        return (
          <div className="title" key={i}>
            <RenderNodes nodes={node.children} />
          </div>
        );
      case "paragraph":
        return (
          <p className="paragraph">
            <RenderNodes nodes={node.children} />
          </p>
        );
      case "title_reference":
        return (
          <span className="title_reference">
            <RenderNodes nodes={node.children} />
          </span>
        );
      case "definition_list":
        return (
          <ul className={node.type}>
            <RenderNodes nodes={node.children} />
          </ul>
        );
      case "definition_list_item":
        return (
          <li className={node.type}>
            <RenderNodes nodes={node.children} />
          </li>
        );
      default:
        return (
          <div className={node.type}>
            <RenderNodes nodes={node.children} />
          </div>
        );
    }
  });
}

function getSignature(name, signature) {
  return `${name}(
    ${signature.slice(1, -1)}
)`;
}
