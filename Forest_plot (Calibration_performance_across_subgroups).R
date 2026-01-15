library(forestplot)
library(grid)
setwd("XXXXXX")#data_path
# Setting Data
df  <- data.frame(
  group = c("XXXX",
            "XXXX", "XXXXX",
            "XXXXX", "XXXX"),
  N = c(XX,XX,XX,XX,XX ),
  O = c(XX, XX, XX, XX,XX ),
  E = c(XX, XX, XX, XX, XX)
)

# Calculate SMR and its 95% CI
df$SMR   <- df$O / df$E
df$lower <- df$SMR * exp(-1.96 / sqrt(df$O))
df$upper <- df$SMR * exp( 1.96 / sqrt(df$O))

# Set table text
tabletext <- cbind(
  c("Group", as.character(df$group)),
  c("N",     as.character(df$N)),
  c("O",     as.character(df$O)),
  c("E",     as.character(df$E)),
  c("SMR (95% CI)", 
    paste0(
      round(df$SMR, 2), 
      " (", 
      round(df$lower, 2), ", ", 
      round(df$upper, 2), 
      ")"
    )
  )
)

mean_vals  <- c(NA, df$SMR)
lower_vals <- c(NA, df$lower)
upper_vals <- c(NA, df$upper)


# Drawing the forest plot
forestplot(
  labeltext = tabletext,
  mean  = mean_vals,
  lower = lower_vals,
  upper = upper_vals,
  zero  = 1,                     
  xlab  = expression("SMR (95% CI)"),  
  title = "Standardized Mortality Ratio (SMR) Forest Plot",
  boxsize   = 0.3,               
  lineheight= unit(1.5, "cm"),   
  col=fpColors(box="royalblue", line="darkblue", summary="darkred"),  
  ci.vertices = TRUE,            
  clip = c(0.5, 2),              
  xticks = seq(0.5, 2, 0.25),   
  new_page = FALSE               
)
# Save as PDF
pdf("Forestplot_SMR.pdf", width = 7, height = 5)  
grid.newpage() 
dev.off()  
