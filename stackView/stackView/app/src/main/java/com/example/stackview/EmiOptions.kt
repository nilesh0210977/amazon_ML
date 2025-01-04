////package com.example.stackview
////
////// File: EmiOptions.kt
////
////import androidx.compose.runtime.*
////import androidx.compose.foundation.layout.*
////import androidx.compose.material3.MaterialTheme
////import androidx.compose.material3.Text
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.unit.dp
////import androidx.compose.ui.unit.sp
////
////@Composable
////fun EmiOptions(items: List<EmiItem>) {
////    Column {
////        items.forEach { emiItem ->
////            Text(
////                text = emiItem.title,
////                fontSize = 14.sp,
////                fontWeight = FontWeight.Bold,
////                modifier = Modifier.padding(8.dp)
////            )
////            Text(text = emiItem.subtitle, color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f))
////        }
////        Text(text = "Create your own plan", color = MaterialTheme.colorScheme.primary)
////    }
////}
//
//
//////chatgpt tried
////package com.example.stackview
////
////import androidx.compose.foundation.background
////import androidx.compose.foundation.layout.*
////import androidx.compose.foundation.shape.RoundedCornerShape
////import androidx.compose.material3.*
////import androidx.compose.runtime.*
////import androidx.compose.ui.Alignment
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.graphics.Color
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.unit.dp
////import androidx.compose.ui.unit.sp
////import androidx.compose.ui.text.style.TextAlign
////import androidx.compose.ui.unit.Dp
////
////@Composable
////fun EmiOptions(items: List<EmiItem>, selectedColor: Color = Color(0xFF4C3A3A)) {
////    Column(
////        modifier = Modifier
////            .fillMaxSize()
////            .background(Color(0xFF1A1A2E))
////            .padding(16.dp)
////    ) {
////        Text(
////            text = "How do you wish to repay?",
////            color = Color.White,
////            fontSize = 18.sp,
////            fontWeight = FontWeight.Bold,
////            modifier = Modifier.padding(bottom = 8.dp)
////        )
////        Text(
////            text = "Choose one of our recommended plans or make your own",
////            color = Color(0xFF9A9A9A),
////            fontSize = 14.sp,
////            modifier = Modifier.padding(bottom = 16.dp)
////        )
////
////        items.forEachIndexed { index, emiItem ->
////            EmiCard(
////                emiItem = emiItem,
////                isSelected = index == 0,
////                selectedColor = selectedColor,
////                modifier = Modifier.padding(bottom = 8.dp)
////            )
////        }
////
////        Spacer(modifier = Modifier.height(16.dp))
////
////        Button(
////            onClick = { /* Handle custom plan creation */ },
////            modifier = Modifier
////                .align(Alignment.CenterHorizontally)
////                .padding(vertical = 8.dp)
////        ) {
////            Text("Create your own plan")
////        }
////
////        Spacer(modifier = Modifier.height(24.dp))
////
////        Text(
////            text = "Select your bank account",
////            color = Color.White,
////            fontSize = 18.sp,
////            textAlign = TextAlign.Center,
////            modifier = Modifier
////                .fillMaxWidth()
////                .background(Color(0xFF5357EA), shape = RoundedCornerShape(8.dp))
////                .padding(vertical = 16.dp)
////        )
////    }
////}
////
////@Composable
////fun EmiCard(emiItem: EmiItem, isSelected: Boolean, selectedColor: Color, modifier: Modifier = Modifier) {
////    val backgroundColor = if (isSelected) selectedColor else Color(0xFF252547)
////    val contentColor = if (isSelected) Color.White else Color(0xFF9A9A9A)
////
////    Box(
////        modifier = modifier
////            .fillMaxWidth()
////            .background(backgroundColor, shape = RoundedCornerShape(8.dp))
////            .padding(16.dp)
////    ) {
////        Column {
////            if (emiItem.tag != null && isSelected) {
////                Box(
////                    modifier = Modifier
////                        .background(Color.White, shape = RoundedCornerShape(12.dp))
////                        .padding(horizontal = 8.dp, vertical = 4.dp)
////                        .align(Alignment.End)
////                ) {
////                    Text(
////                        text = emiItem.tag,
////                        color = selectedColor,
////                        fontSize = 12.sp,
////                        fontWeight = FontWeight.Bold
////                    )
////                }
////            }
////
////            Text(
////                text = emiItem.emi,
////                color = contentColor,
////                fontSize = 20.sp,
////                fontWeight = FontWeight.Bold
////            )
////            Text(
////                text = emiItem.duration,
////                color = contentColor,
////                fontSize = 14.sp
////            )
////            Spacer(modifier = Modifier.height(4.dp))
////            Text(
////                text = emiItem.subtitle,
////                color = contentColor,
////                fontSize = 12.sp
////            )
////        }
////    }
////}
//
//
//
////try with carousel
////package com.example.stackview
////
////import androidx.compose.animation.core.animateFloatAsState
////import androidx.compose.foundation.ExperimentalFoundationApi
////import androidx.compose.foundation.background
////import androidx.compose.foundation.layout.*
////import androidx.compose.foundation.pager.HorizontalPager
////import androidx.compose.foundation.pager.PagerState
////import androidx.compose.foundation.pager.rememberPagerState
////import androidx.compose.material3.*
////import androidx.compose.runtime.*
////import androidx.compose.ui.Alignment
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.graphics.Color
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.text.style.TextAlign
////import androidx.compose.ui.unit.dp
////import kotlinx.coroutines.launch
////
//////data class EmiItem(
//////    val emi: String,
//////    val duration: String,
//////    val subtitle: String,
//////    val tag: String? = null
//////)
////
////@OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)
////@Composable
////fun EmiOptions(
////    items: List<EmiItem>,
////    selectedColor: Color = MaterialTheme.colorScheme.primaryContainer,
////    onCreatePlanClick: () -> Unit = {},
////    onBankSelectionClick: () -> Unit = {}
////) {
////    val pagerState = rememberPagerState(pageCount = { items.size })
////    val scope = rememberCoroutineScope()
////
////    Surface(
////        modifier = Modifier.fillMaxSize(),
////        color = MaterialTheme.colorScheme.background
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .padding(16.dp)
////        ) {
////            // Header Section
////            Text(
////                text = "How do you wish to repay?",
////                style = MaterialTheme.typography.titleLarge,
////                color = MaterialTheme.colorScheme.onBackground
////            )
////
////            Text(
////                text = "Choose one of our recommended plans or make your own",
////                style = MaterialTheme.typography.bodyMedium,
////                color = MaterialTheme.colorScheme.onSurfaceVariant,
////                modifier = Modifier.padding(top = 8.dp, bottom = 16.dp)
////            )
////
////            // EMI Options Carousel
////            Box(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .height(240.dp)
////            ) {
////                HorizontalPager(
////                    state = pagerState,
////                    modifier = Modifier.fillMaxWidth()
////                ) { page ->
////                    EmiCard(
////                        emiItem = items[page],
////                        isSelected = pagerState.currentPage == page,
////                        selectedColor = selectedColor,
////                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
////                    )
////                }
////            }
////
////            // Pager Indicators
////            PagerIndicator(
////                pagerState = pagerState,
////                modifier = Modifier
////                    .align(Alignment.CenterHorizontally)
////                    .padding(16.dp)
////            )
////
////            Spacer(modifier = Modifier.height(16.dp))
////
////            // Create Plan Button
////            FilledTonalButton(
////                onClick = onCreatePlanClick,
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(vertical = 8.dp)
////            ) {
////                Text(
////                    text = "Create your own plan",
////                    style = MaterialTheme.typography.labelLarge
////                )
////            }
////
////            Spacer(modifier = Modifier.height(24.dp))
////
////            // Bank Selection Card
////            Card(
////                onClick = onBankSelectionClick,
////                colors = CardDefaults.cardColors(
////                    containerColor = MaterialTheme.colorScheme.primaryContainer
////                ),
////                modifier = Modifier.fillMaxWidth()
////            ) {
////                Text(
////                    text = "Select your bank account",
////                    style = MaterialTheme.typography.titleMedium,
////                    textAlign = TextAlign.Center,
////                    modifier = Modifier
////                        .fillMaxWidth()
////                        .padding(vertical = 16.dp)
////                )
////            }
////        }
////    }
////}
////
////@Composable
////fun EmiCard(
////    emiItem: EmiItem,
////    isSelected: Boolean,
////    selectedColor: Color,
////    modifier: Modifier = Modifier
////) {
////    Card(
////        modifier = modifier.fillMaxWidth(),
////        colors = CardDefaults.cardColors(
////            containerColor = if (isSelected)
////                selectedColor
////            else
////                MaterialTheme.colorScheme.surfaceVariant
////        ),
////        elevation = CardDefaults.cardElevation(
////            defaultElevation = if (isSelected) 8.dp else 1.dp
////        )
////    ) {
////        Column(
////            modifier = Modifier.padding(16.dp)
////        ) {
////            if (emiItem.tag != null && isSelected) {
////                Card(
////                    colors = CardDefaults.cardColors(
////                        containerColor = MaterialTheme.colorScheme.surface
////                    ),
////                    modifier = Modifier.align(Alignment.End)
////                ) {
////                    Text(
////                        text = emiItem.tag,
////                        style = MaterialTheme.typography.labelSmall,
////                        color = MaterialTheme.colorScheme.primary,
////                        modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
////                    )
////                }
////            }
////
////            Text(
////                text = emiItem.emi,
////                style = MaterialTheme.typography.headlineSmall.copy(
////                    fontWeight = FontWeight.Bold
////                ),
////                color = if (isSelected)
////                    MaterialTheme.colorScheme.onPrimaryContainer
////                else
////                    MaterialTheme.colorScheme.onSurfaceVariant
////            )
////
////            Text(
////                text = emiItem.duration,
////                style = MaterialTheme.typography.bodyLarge,
////                color = if (isSelected)
////                    MaterialTheme.colorScheme.onPrimaryContainer
////                else
////                    MaterialTheme.colorScheme.onSurfaceVariant
////            )
////
////            Spacer(modifier = Modifier.height(4.dp))
////
////            Text(
////                text = emiItem.subtitle,
////                style = MaterialTheme.typography.bodyMedium,
////                color = if (isSelected)
////                    MaterialTheme.colorScheme.onPrimaryContainer
////                else
////                    MaterialTheme.colorScheme.onSurfaceVariant
////            )
////        }
////    }
////}
////
////@OptIn(ExperimentalFoundationApi::class)
////@Composable
////private fun PagerIndicator(
////    pagerState: PagerState,
////    modifier: Modifier = Modifier
////) {
////    Row(
////        horizontalArrangement = Arrangement.Center,
////        modifier = modifier
////    ) {
////        repeat(pagerState.pageCount) { iteration ->
////            val color = if (pagerState.currentPage == iteration)
////                MaterialTheme.colorScheme.primary
////            else
////                MaterialTheme.colorScheme.surfaceVariant
////
////            Box(
////                modifier = Modifier
////                    .padding(2.dp)
////                    .size(8.dp)
////                    .surface(
////                        shape = MaterialTheme.shapes.small,
////                        color = color
////                    )
////            )
////        }
////    }
////}
////
////// Extension function to create a surface with specific shape and color
////private fun Modifier.surface(
////    shape: androidx.compose.ui.graphics.Shape,
////    color: Color
////) = this.then(
////    Modifier
////        .background(
////            color = color,
////            shape = shape
////        )
////)
//
//
////designated changes
////
////package com.example.stackview
////
////import androidx.compose.foundation.BorderStroke
////import androidx.compose.foundation.ExperimentalFoundationApi
////import androidx.compose.foundation.layout.*
////import androidx.compose.foundation.shape.RoundedCornerShape
////import androidx.compose.foundation.pager.HorizontalPager
////import androidx.compose.foundation.pager.rememberPagerState
////import androidx.compose.foundation.shape.CircleShape
////import androidx.compose.material3.*
////import androidx.compose.runtime.*
////import androidx.compose.ui.Alignment
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.graphics.Color
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.text.style.TextAlign
////import androidx.compose.ui.unit.dp
////import androidx.compose.ui.unit.sp
////
////@OptIn(ExperimentalFoundationApi::class)
////@Composable
////fun EmiOptions(
////    items: List<EmiItem>,
////    modifier: Modifier = Modifier,
////    onCreatePlanClick: () -> Unit = {},
////    onBankSelectionClick: () -> Unit = {}
////) {
////    val pagerState = rememberPagerState(pageCount = { items.size })
////
////    Surface(
////        modifier = modifier.fillMaxSize(),
////        color = Color(0xFF0A0A1A)
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxWidth()
////                .padding(20.dp)
////        ) {
////            // EMI Cards Pager
////            HorizontalPager(
////                state = pagerState,
////                contentPadding = PaddingValues(horizontal = 40.dp),
////                pageSpacing = 16.dp,
////                modifier = Modifier.height(200.dp) // Increased height for better visibility
////            ) { page ->
////                EmiCard(
////                    emiItem = items[page],
////                    isSelected = pagerState.currentPage == page,
////                    modifier = Modifier.fillMaxWidth()
////                )
////            }
////
////            // Create your own plan button
////            Button(
////                onClick = onCreatePlanClick,
////                colors = ButtonDefaults.buttonColors(
////                    containerColor = Color(0xFF1F1F35),
////                    contentColor = Color(0xFF8E8EA8)
////                ),
////                shape = RoundedCornerShape(12.dp),
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(vertical = 24.dp)
////            ) {
////                Text(
////                    text = "Create your own plan",
////                    fontSize = 15.sp,
////                    modifier = Modifier.padding(vertical = 4.dp)
////                )
////            }
////
////            Spacer(modifier = Modifier.weight(1f))
////
////            // Bank account selection button
////            Button(
////                onClick = onBankSelectionClick,
////                colors = ButtonDefaults.buttonColors(
////                    containerColor = Color(0xFF5458EA)
////                ),
////                shape = RoundedCornerShape(12.dp),
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 16.dp)
////            ) {
////                Text(
////                    text = "Select your bank account",
////                    fontSize = 16.sp,
////                    modifier = Modifier.padding(vertical = 8.dp)
////                )
////            }
////        }
////    }
////}
////
////@Composable
////fun EmiCard(
////    emiItem: EmiItem,
////    isSelected: Boolean,
////    modifier: Modifier = Modifier
////) {
////    Card(
////        shape = RoundedCornerShape(16.dp),
////        colors = CardDefaults.cardColors(
////            containerColor = if (isSelected) Color(0xFF272741) else Color(0xFF1F1F35)
////        ),
////        modifier = modifier
////            .height(200.dp) // Increased height for better visibility
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .padding(16.dp),
////            horizontalAlignment = Alignment.CenterHorizontally
////        ) {
////            // Top row with checkbox and recommended tag
////            Row(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 16.dp),
////                horizontalArrangement = Arrangement.SpaceBetween,
////                verticalAlignment = Alignment.CenterVertically
////            ) {
////                // Checkbox (filled or unfilled)
////                Card(
////                    colors = CardDefaults.cardColors(
////                        containerColor = if (isSelected) Color(0xFF5458EA) else Color(0xFF1F1F35)
////                    ),
////                    border = if (!isSelected) BorderStroke(1.dp, Color(0xFF5458EA)) else null,
////                    shape = CircleShape,
////                    modifier = Modifier.size(24.dp)
////                ) {
////                    if (isSelected) {
////                        Box(
////                            contentAlignment = Alignment.Center,
////                            modifier = Modifier.fillMaxSize()
////                        ) {
////                            Text(
////                                text = "✓",
////                                color = Color.White,
////                                fontSize = 14.sp
////                            )
////                        }
////                    }
////                }
////
////                // Recommended tag if present
////                emiItem.tag?.let { tag ->
////                    Card(
////                        colors = CardDefaults.cardColors(
////                            containerColor = Color.White
////                        ),
////                        shape = RoundedCornerShape(8.dp)
////                    ) {
////                        Text(
////                            text = tag,
////                            color = Color(0xFF5458EA),
////                            fontSize = 12.sp,
////                            fontWeight = FontWeight.Medium,
////                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
////                        )
////                    }
////                }
////            }
////
////            // EMI amount
////            Row(
////                verticalAlignment = Alignment.Bottom,
////                modifier = Modifier.padding(bottom = 4.dp)
////            ) {
////                Text(
////                    text = emiItem.emi,
////                    color = Color.White,
////                    fontSize = 24.sp,
////                    fontWeight = FontWeight.SemiBold
////                )
////
////                Text(
////                    text = "/mo",
////                    color = Color(0xFF9A9A9A),
////                    fontSize = 16.sp,
////                    modifier = Modifier.padding(start = 4.dp, bottom = 2.dp)
////                )
////            }
////
////            // Duration text
////            Text(
////                text = emiItem.duration,
////                color = Color(0xFF9A9A9A),
////                fontSize = 14.sp
////            )
////
////            // Subtitle ("See calculations")
////            Text(
////                text = emiItem.subtitle,
////                color = Color(0xFF5458EA),
////                fontSize = 14.sp,
////                modifier = Modifier.padding(top = 8.dp)
////            )
////        }
////    }
////}
////
////// Example usage:
////@Composable
////fun PreviewEmiOptions() {
////    val sampleItems = listOf(
////        EmiItem(
////            emi = "₹4,247",
////            duration = "for 12 months",
////            title = "12 Months Plan",
////            subtitle = "See calculations"
////        ),
////        EmiItem(
////            emi = "₹5,580",
////            duration = "for 9 months",
////            title = "9 Months Plan",
////            subtitle = "See calculations",
////            tag = "recommended"
////        ),
////        EmiItem(
////            emi = "₹8,247",
////            duration = "for 6 months",
////            title = "6 Months Plan",
////            subtitle = "See calculations"
////        )
////    )
////
////    EmiOptions(items = sampleItems)
////}
//
//
//
////gpt final hustel
//
////
////package com.example.stackview
////
////import androidx.compose.foundation.BorderStroke
////import androidx.compose.foundation.ExperimentalFoundationApi
////import androidx.compose.foundation.clickable
////import androidx.compose.foundation.layout.*
////import androidx.compose.foundation.pager.HorizontalPager
////import androidx.compose.foundation.pager.rememberPagerState
////import androidx.compose.foundation.shape.CircleShape
////import androidx.compose.foundation.shape.RoundedCornerShape
////import androidx.compose.material3.*
////import androidx.compose.runtime.*
////import androidx.compose.ui.Alignment
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.graphics.Color
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.unit.dp
////import androidx.compose.ui.unit.sp
////
////@OptIn(ExperimentalFoundationApi::class)
////@Composable
////fun EmiOptions(
////    items: List<EmiItem>,
////    modifier: Modifier = Modifier,
////    onCreatePlanClick: () -> Unit = {},
////    onBankSelectionClick: () -> Unit = {}
////) {
////    val pagerState = rememberPagerState(pageCount = { items.size })
////    var selectedCardIndex by remember { mutableStateOf(0) }
////
////    Surface(
////        modifier = modifier.fillMaxSize(),
////        color = Color(0xFF0A0A1A)
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxWidth()
////                .padding(20.dp)
////        ) {
////            // EMI Cards Pager
////            HorizontalPager(
////                state = pagerState,
////                contentPadding = PaddingValues(horizontal = 40.dp),
////                pageSpacing = 16.dp,
////                modifier = Modifier.height(200.dp) // Increased height for better visibility
////            ) { page ->
////                EmiCard(
////                    emiItem = items[page],
////                    isSelected = selectedCardIndex == page,
////                    modifier = Modifier
////                        .fillMaxWidth()
////                        .clickable {
////                            selectedCardIndex = page // Update selected index on click
////                        }
////                )
////            }
////
////            // Create your own plan button
////            Button(
////                onClick = onCreatePlanClick,
////                colors = ButtonDefaults.buttonColors(
////                    containerColor = Color(0xFF1F1F35),
////                    contentColor = Color(0xFF8E8EA8)
////                ),
////                shape = RoundedCornerShape(12.dp),
////                modifier = Modifier
////                    .wrapContentWidth() // Button width only matches the text width
////                    .align(Alignment.Start) // Align the button to the left
////                    .padding(vertical = 24.dp)
////            ) {
////                Text(
////                    text = "Create your own plan",
////                    fontSize = 15.sp,
////                    modifier = Modifier.padding(vertical = 4.dp)
////                )
////            }
////
////            Spacer(modifier = Modifier.weight(1f))
////
////            // Bank account selection button
////            Button(
////                onClick = onBankSelectionClick,
////                colors = ButtonDefaults.buttonColors(
////                    containerColor = Color(0xFF5458EA)
////                ),
////                shape = RoundedCornerShape(12.dp),
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 16.dp)
////            ) {
////                Text(
////                    text = "Select your bank account",
////                    fontSize = 16.sp,
////                    modifier = Modifier.padding(vertical = 8.dp)
////                )
////            }
////        }
////    }
////}
////
////@Composable
////fun EmiCard(
////    emiItem: EmiItem,
////    isSelected: Boolean,
////    modifier: Modifier = Modifier
////) {
////    Card(
////        shape = RoundedCornerShape(16.dp),
////        colors = CardDefaults.cardColors(
////            containerColor = if (isSelected) Color(0xFF272741) else Color(0xFF1F1F35)
////        ),
////        modifier = modifier
////            .height(200.dp) // Increased height for better visibility
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .padding(16.dp),
////            horizontalAlignment = Alignment.CenterHorizontally
////        ) {
////            // Top row with checkbox and recommended tag
////            Row(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 16.dp),
////                horizontalArrangement = Arrangement.SpaceBetween,
////                verticalAlignment = Alignment.CenterVertically
////            ) {
////                // Checkbox (filled or unfilled)
////                Card(
////                    colors = CardDefaults.cardColors(
////                        containerColor = if (isSelected) Color(0xFF5458EA) else Color(0xFF1F1F35)
////                    ),
////                    border = if (!isSelected) BorderStroke(1.dp, Color(0xFF5458EA)) else null,
////                    shape = CircleShape,
////                    modifier = Modifier.size(24.dp)
////                ) {
////                    if (isSelected) {
////                        Box(
////                            contentAlignment = Alignment.Center,
////                            modifier = Modifier.fillMaxSize()
////                        ) {
////                            Text(
////                                text = "✓",
////                                color = Color.White,
////                                fontSize = 14.sp
////                            )
////                        }
////                    }
////                }
////
////                // Recommended tag if present
////                emiItem.tag?.let { tag ->
////                    Card(
////                        colors = CardDefaults.cardColors(
////                            containerColor = Color.White
////                        ),
////                        shape = RoundedCornerShape(8.dp)
////                    ) {
////                        Text(
////                            text = tag,
////                            color = Color(0xFF5458EA),
////                            fontSize = 12.sp,
////                            fontWeight = FontWeight.Medium,
////                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
////                        )
////                    }
////                }
////            }
////
////            // EMI amount
////            Row(
////                verticalAlignment = Alignment.Bottom,
////                modifier = Modifier.padding(bottom = 4.dp)
////            ) {
////                Text(
////                    text = emiItem.emi,
////                    color = Color.White,
////                    fontSize = 24.sp,
////                    fontWeight = FontWeight.SemiBold
////                )
////
////                Text(
////                    text = "/mo",
////                    color = Color(0xFF9A9A9A),
////                    fontSize = 16.sp,
////                    modifier = Modifier.padding(start = 4.dp, bottom = 2.dp)
////                )
////            }
////
////            // Duration text
////            Text(
////                text = emiItem.duration,
////                color = Color(0xFF9A9A9A),
////                fontSize = 14.sp
////            )
////
////            // Subtitle ("See calculations")
////            Text(
////                text = emiItem.subtitle,
////                color = Color(0xFF5458EA),
////                fontSize = 14.sp,
////                modifier = Modifier.padding(top = 8.dp)
////            )
////        }
////    }
////}
////
////// Example usage:
////@Composable
////fun PreviewEmiOptions() {
////    val sampleItems = listOf(
////        EmiItem(
////            emi = "₹4,247",
////            duration = "for 12 months",
////            title = "12 Months Plan",
////            subtitle = "See calculations"
////        ),
////        EmiItem(
////            emi = "₹5,580",
////            duration = "for 9 months",
////            title = "9 Months Plan",
////            subtitle = "See calculations",
////            tag = "recommended"
////        ),
////        EmiItem(
////            emi = "₹8,247",
////            duration = "for 6 months",
////            title = "6 Months Plan",
////            subtitle = "See calculations"
////        )
////    )
////
////    EmiOptions(items = sampleItems)
////}
//
////
////package com.example.stackview
////
////import androidx.compose.foundation.BorderStroke
////import androidx.compose.foundation.ExperimentalFoundationApi
////import androidx.compose.foundation.clickable
////import androidx.compose.foundation.layout.*
////import androidx.compose.foundation.pager.HorizontalPager
////import androidx.compose.foundation.pager.rememberPagerState
////import androidx.compose.foundation.shape.CircleShape
////import androidx.compose.foundation.shape.RoundedCornerShape
////import androidx.compose.material3.*
////import androidx.compose.runtime.*
////import androidx.compose.ui.Alignment
////import androidx.compose.ui.Modifier
////import androidx.compose.ui.graphics.Color
////import androidx.compose.ui.text.font.FontWeight
////import androidx.compose.ui.unit.dp
////import androidx.compose.ui.unit.sp
////
////@OptIn(ExperimentalFoundationApi::class)
////@Composable
////fun EmiOptions(
////    items: List<EmiItem>,
////    modifier: Modifier = Modifier,
////    onCreatePlanClick: () -> Unit = {}
////) {
////    val pagerState = rememberPagerState(pageCount = { items.size })
////    var selectedCardIndex by remember { mutableStateOf(0) }
////
////    Column(
////        modifier = Modifier
////            .fillMaxWidth()
////            .padding(horizontal = 16.dp, vertical = 8.dp)
////    ) {
////        // EMI Cards Pager
////        HorizontalPager(
////            state = pagerState,
////            contentPadding = PaddingValues(horizontal = 40.dp),
////            pageSpacing = 16.dp,
////            modifier = Modifier.height(220.dp)
////        ) { page ->
////            EmiCard(
////                emiItem = items[page],
////                isSelected = selectedCardIndex == page,
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .clickable {
////                        selectedCardIndex = page
////                    }
////            )
////        }
////
////        // Create your own plan button
////        Button(
////            onClick = onCreatePlanClick,
////            colors = ButtonDefaults.buttonColors(
////                containerColor = Color(0xFF1F1F35),
////                contentColor = Color(0xFF8E8EA8)
////            ),
////            shape = RoundedCornerShape(12.dp),
////            modifier = Modifier
////                .wrapContentWidth()
////                .align(Alignment.Start)
////                .padding(vertical = 24.dp)
////        ) {
////            Text(
////                text = "Create your own plan",
////                fontSize = 15.sp,
////                fontWeight = FontWeight.SemiBold,
////                modifier = Modifier.padding(vertical = 4.dp)
////            )
////        }
////
////        Spacer(modifier = Modifier.height(16.dp))
////
////        // Select bank account button (leave navigation handling to StackCard or parent)
////        Button(
////            onClick = { /* Define click action in StackCard or parent composable */ },
////            colors = ButtonDefaults.buttonColors(
////                containerColor = Color(0xFF5458EA)
////            ),
////            shape = RoundedCornerShape(12.dp),
////            modifier = Modifier.fillMaxWidth()
////        ) {
////            Text(
////                text = "Select your bank account",
////                fontSize = 16.sp,
////                fontWeight = FontWeight.Bold,
////                modifier = Modifier.padding(vertical = 8.dp)
////            )
////        }
////    }
////}
////
////@Composable
////fun EmiCard(
////    emiItem: EmiItem,
////    isSelected: Boolean,
////    modifier: Modifier = Modifier
////) {
////    Card(
////        shape = RoundedCornerShape(16.dp),
////        colors = CardDefaults.cardColors(
////            containerColor = if (isSelected) Color(0xFF272741) else Color(0xFF1F1F35)
////        ),
////        modifier = modifier
////            .height(200.dp)
////            .padding(8.dp)
////    ) {
////        Column(
////            modifier = Modifier
////                .fillMaxSize()
////                .padding(16.dp),
////            horizontalAlignment = Alignment.CenterHorizontally
////        ) {
////            // Top row with checkbox and recommended tag
////            Row(
////                modifier = Modifier
////                    .fillMaxWidth()
////                    .padding(bottom = 16.dp),
////                horizontalArrangement = Arrangement.SpaceBetween,
////                verticalAlignment = Alignment.CenterVertically
////            ) {
////                // Checkbox
////                Card(
////                    colors = CardDefaults.cardColors(
////                        containerColor = if (isSelected) Color(0xFF5458EA) else Color(0xFF1F1F35)
////                    ),
////                    border = if (!isSelected) BorderStroke(1.dp, Color(0xFF5458EA)) else null,
////                    shape = CircleShape,
////                    modifier = Modifier.size(24.dp)
////                ) {
////                    if (isSelected) {
////                        Box(
////                            contentAlignment = Alignment.Center,
////                            modifier = Modifier.fillMaxSize()
////                        ) {
////                            Text(
////                                text = "✓",
////                                color = Color.White,
////                                fontSize = 14.sp
////                            )
////                        }
////                    }
////                }
////
////                // Recommended tag if present
////                emiItem.tag?.let { tag ->
////                    Card(
////                        colors = CardDefaults.cardColors(
////                            containerColor = Color.White
////                        ),
////                        shape = RoundedCornerShape(8.dp)
////                    ) {
////                        Text(
////                            text = tag,
////                            color = Color(0xFF5458EA),
////                            fontSize = 12.sp,
////                            fontWeight = FontWeight.Medium,
////                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
////                        )
////                    }
////                }
////            }
////
////            // EMI amount
////            Text(
////                text = "${emiItem.emi}",
////                color = Color.White,
////                fontSize = 24.sp,
////                fontWeight = FontWeight.SemiBold,
////                modifier = Modifier.padding(bottom = 4.dp)
////            )
////
////            // Duration text
////            Text(
////                text = emiItem.duration,
////                color = Color(0xFF9A9A9A),
////                fontSize = 14.sp
////            )
////
////            // Subtitle ("See calculations")
////            Text(
////                text = emiItem.subtitle,
////                color = Color(0xFF5458EA),
////                fontSize = 14.sp,
////                modifier = Modifier.padding(top = 8.dp)
////            )
////        }
////    }
////}
////
////// Data class for EmiItem
//
//
//
////final emioptions using navigation
//
//package com.example.stackview
//
//import android.service.autofill.OnClickAction
//import androidx.compose.foundation.BorderStroke
//import androidx.compose.foundation.ExperimentalFoundationApi
//import androidx.compose.foundation.clickable
//import androidx.compose.foundation.layout.*
//import androidx.compose.foundation.pager.HorizontalPager
//import androidx.compose.foundation.pager.rememberPagerState
//import androidx.compose.foundation.shape.CircleShape
//import androidx.compose.foundation.shape.RoundedCornerShape
//import androidx.compose.material3.*
//import androidx.compose.runtime.*
//import androidx.compose.ui.Alignment
//import androidx.compose.ui.Modifier
//import androidx.compose.ui.graphics.Color
//import androidx.compose.ui.text.font.FontWeight
//import androidx.compose.ui.unit.dp
//import androidx.compose.ui.unit.sp
//
//@OptIn(ExperimentalFoundationApi::class)
//@Composable
//fun EmiOptions(
//    items: List<EmiItem>,
//    modifier: Modifier = Modifier,
//    onCreatePlanClick: () -> Unit = {},
//    onSelectBankClick: () -> Unit = {} // Navigation handler added here
//) {
//    val pagerState = rememberPagerState(pageCount = { items.size })
//    var selectedCardIndex by remember { mutableStateOf(0) }
//
//    Column(
//        modifier = Modifier
//            .fillMaxWidth()
//            .padding(horizontal = 16.dp, vertical = 8.dp)
//    ) {
//        HorizontalPager(
//            state = pagerState,
//            contentPadding = PaddingValues(horizontal = 40.dp),
//            pageSpacing = 16.dp,
//            modifier = Modifier.height(220.dp)
//        ) { page ->
//            EmiCard(
//                emiItem = items[page],
//                isSelected = selectedCardIndex == page,
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .clickable {
//                        selectedCardIndex = page
//                    }
//            )
//        }
//
//        Button(
//            onClick = onCreatePlanClick,
//            colors = ButtonDefaults.buttonColors(
//                containerColor = Color(0xFF1F1F35),
//                contentColor = Color(0xFF8E8EA8)
//            ),
//            shape = RoundedCornerShape(12.dp),
//            modifier = Modifier
//                .wrapContentWidth()
//                .align(Alignment.Start)
//                .padding(vertical = 24.dp)
//        ) {
//            Text(
//                text = "Create your own plan",
//                fontSize = 15.sp,
//                fontWeight = FontWeight.SemiBold,
//                modifier = Modifier.padding(vertical = 4.dp)
//            )
//        }
//
//        Spacer(modifier = Modifier.height(16.dp))
//
//        Button(
//            onClick = onSelectBankClick, // Trigger navigation here
//            colors = ButtonDefaults.buttonColors(
//                containerColor = Color(0xFF5458EA)
//            ),
//            shape = RoundedCornerShape(12.dp),
//            modifier = Modifier.fillMaxWidth()
//        ) {
//            Text(
//                text = "Select your bank account",
//
//                fontSize = 16.sp,
//                fontWeight = FontWeight.Bold,
//                modifier = Modifier.padding(vertical = 8.dp)
//            )
//        }
//    }
//}
//
//@Composable
//fun EmiCard(
//    emiItem: EmiItem,
//    isSelected: Boolean,
//    modifier: Modifier = Modifier
//) {
//    Card(
//        shape = RoundedCornerShape(16.dp),
//        colors = CardDefaults.cardColors(
//            containerColor = if (isSelected) Color(0xFF272741) else Color(0xFF1F1F35)
//        ),
//        modifier = modifier
//            .height(200.dp)
//            .padding(8.dp)
//    ) {
//        Column(
//            modifier = Modifier
//                .fillMaxSize()
//                .padding(16.dp),
//            horizontalAlignment = Alignment.CenterHorizontally
//        ) {
//            Row(
//                modifier = Modifier
//                    .fillMaxWidth()
//                    .padding(bottom = 16.dp),
//                horizontalArrangement = Arrangement.SpaceBetween,
//                verticalAlignment = Alignment.CenterVertically
//            ) {
//                Card(
//                    colors = CardDefaults.cardColors(
//                        containerColor = if (isSelected) Color(0xFF5458EA) else Color(0xFF1F1F35)
//                    ),
//                    border = if (!isSelected) BorderStroke(1.dp, Color(0xFF5458EA)) else null,
//                    shape = CircleShape,
//                    modifier = Modifier.size(24.dp)
//                ) {
//                    if (isSelected) {
//                        Box(
//                            contentAlignment = Alignment.Center,
//                            modifier = Modifier.fillMaxSize()
//                        ) {
//                            Text(
//                                text = "✓",
//                                color = Color.White,
//                                fontSize = 14.sp
//                            )
//                        }
//                    }
//                }
//
//                emiItem.tag?.let { tag ->
//                    Card(
//                        colors = CardDefaults.cardColors(
//                            containerColor = Color.White
//                        ),
//                        shape = RoundedCornerShape(8.dp)
//                    ) {
//                        Text(
//                            text = tag,
//                            color = Color(0xFF5458EA),
//                            fontSize = 12.sp,
//                            fontWeight = FontWeight.Medium,
//                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
//                        )
//                    }
//                }
//            }
//
//            Text(
//                text = "${emiItem.emi}",
//                color = Color.White,
//                fontSize = 24.sp,
//                fontWeight = FontWeight.SemiBold,
//                modifier = Modifier.padding(bottom = 4.dp)
//            )
//
//            Text(
//                text = emiItem.duration,
//                color = Color(0xFF9A9A9A),
//                fontSize = 14.sp
//            )
//
//            Text(
//                text = emiItem.subtitle,
//                color = Color(0xFF5458EA),
//                fontSize = 14.sp,
//                modifier = Modifier.padding(top = 8.dp)
//            )
//        }
//    }
//}
//
//// Data class for EmiItem
//
//

package com.example.stackview

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@OptIn(ExperimentalFoundationApi::class)
@Composable
fun EmiOptions(
    items: List<EmiItem>,
    modifier: Modifier = Modifier,
    onCreatePlanClick: () -> Unit = {},
    onSelectBankClick: () -> Unit = {} // Navigation handler added here
) {
    val pagerState = rememberPagerState(pageCount = { items.size })
    var selectedCardIndex by remember { mutableStateOf(0) }
    var isExpanded by remember { mutableStateOf(false) } // Track if the back option stack is expanded

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 8.dp)
    ) {
        if (isExpanded) {
            // Expanded back stack with additional options
            Text(
                text = "Bank Account Options",
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = Color.Black,
                modifier = Modifier.padding(bottom = 16.dp)
            )

            Button(
                onClick = { isExpanded = false }, // Collapse on back click
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.Gray
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Back",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        } else {
            // Standard EMI options
            HorizontalPager(
                state = pagerState,
                contentPadding = PaddingValues(horizontal = 40.dp),
                pageSpacing = 16.dp,
                modifier = Modifier.height(220.dp)
            ) { page ->
                EmiCard(
                    emiItem = items[page],
                    isSelected = selectedCardIndex == page,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            selectedCardIndex = page
                        }
                )
            }

            Button(
                onClick = onCreatePlanClick,
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF1F1F35),
                    contentColor = Color(0xFF8E8EA8)
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .wrapContentWidth()
                    .align(Alignment.Start)
                    .padding(vertical = 24.dp)
            ) {
                Text(
                    text = "Create your own plan",
                    fontSize = 15.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(
                onClick = { isExpanded = true }, // Expand back option stack here
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color(0xFF5458EA)
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = "Select your bank account",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.padding(vertical = 8.dp)
                )
            }
        }
    }
}

@Composable
fun EmiCard(
    emiItem: EmiItem,
    isSelected: Boolean,
    modifier: Modifier = Modifier
) {
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (isSelected) Color(0xFF272741) else Color(0xFF1F1F35)
        ),
        modifier = modifier
            .height(200.dp)
            .padding(8.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (isSelected) Color(0xFF5458EA) else Color(0xFF1F1F35)
                    ),
                    border = if (!isSelected) BorderStroke(1.dp, Color(0xFF5458EA)) else null,
                    shape = CircleShape,
                    modifier = Modifier.size(24.dp)
                ) {
                    if (isSelected) {
                        Box(
                            contentAlignment = Alignment.Center,
                            modifier = Modifier.fillMaxSize()
                        ) {
                            Text(
                                text = "✓",
                                color = Color.White,
                                fontSize = 14.sp
                            )
                        }
                    }
                }

                emiItem.tag?.let { tag ->
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = Color.White
                        ),
                        shape = RoundedCornerShape(8.dp)
                    ) {
                        Text(
                            text = tag,
                            color = Color(0xFF5458EA),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Medium,
                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp)
                        )
                    }
                }
            }

            Text(
                text = "${emiItem.emi}",
                color = Color.White,
                fontSize = 24.sp,
                fontWeight = FontWeight.SemiBold,
                modifier = Modifier.padding(bottom = 4.dp)
            )

            Text(
                text = emiItem.duration,
                color = Color(0xFF9A9A9A),
                fontSize = 14.sp
            )

            Text(
                text = emiItem.subtitle,
                color = Color(0xFF5458EA),
                fontSize = 14.sp,
                modifier = Modifier.padding(top = 8.dp)
            )
        }
    }
}

// Data class for EmiItem
