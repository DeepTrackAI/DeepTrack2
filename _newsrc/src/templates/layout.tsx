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
  Link,
} from "@mui/material";

import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import Sidebar from "./sidebar";

// nightOwl

export default function Layout({ children, pageContext: { modules, module } }) {
  console.log(modules);

  const initialState: Record<string, any> = {};
  Object.keys(modules).forEach((key) => {
    initialState[key] = key === module;
  });
  const [open, setOpen] = React.useState(initialState);

  const handleClick = (key) => () => {
    setOpen({ ...open, [key]: !open[key] });
  };

  const drawerWidth = 300;

  return (
    <Box sx={{ display: "flex" }}>
      <AppBar
        position="fixed"
        sx={{ zIndex: (theme) => theme.zIndex.drawer + 1, background: "white" }}
      >
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Clipped drawer
          </Typography>
        </Toolbar>
      </AppBar>

      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: {
            width: drawerWidth,
            boxSizing: "border-box",
          },
        }}
      >
        <Toolbar />
        <Box sx={{ overflow: "auto" }}>
          {Object.keys(modules).map((key, j) => (
            <React.Fragment key={j}>
              <ListItemButton key={j} onClick={handleClick(key)}>
                <ListItemText primary={key.replace("deeptrack.", "")} />
                {open[key] ? <ExpandLess /> : <ExpandMore />}
              </ListItemButton>
              <Collapse key={key} in={open[key]} timeout="auto" unmountOnExit>
                <List component="div" disablePadding>
                  {modules[key].map((name, i) => (
                    <ListItem
                      button
                      component={Link}
                      href={`../${key}/${name}`}
                      sx={{ pl: 4 }}
                      key={i}
                    >
                      <ListItemText primary={name} />
                    </ListItem>
                  ))}
                </List>
              </Collapse>
            </React.Fragment>
          ))}
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

function RenderNodes({ nodes }) {
  console.log(nodes);
  return nodes.map((node, i) => {
    if (node.type === "Text") {
      return (
        <span className={node.type} key={i}>
          {node.content}
        </span>
      );
    }

    console.log(node, node.type);

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
