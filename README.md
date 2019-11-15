# Data-Science-Applications
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "=====================================================================\n",
      "Assignment: Project 2: Inference and Capital Punishment\n",
      "OK, version v1.12.5\n",
      "=====================================================================\n",
      "\n",
      "Successfully logged in as sarahtrefler@berkeley.edu\n"
     ]
    }
   ],
   "source": [
    "from datascience import *\n",
    "import numpy as np\n",
    "\n",
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plots\n",
    "plots.style.use('fivethirtyeight')\n",
    "\n",
    "from client.api.notebook import Notebook\n",
    "ok = Notebook('project2.ok')\n",
    "_ = ok.auth(inline=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Overview\n",
    "\n",
    "Punishment for crime has many [philosophical justifications](http://plato.stanford.edu/entries/punishment/#ThePun).  An important one is that fear of punishment may *deter* people from committing crimes.\n",
    "\n",
    "In the United States, some jurisdictions execute some people who are convicted of particularly serious crimes, such as murder.  This punishment is called the *death penalty* or *capital punishment*.  The death penalty is controversial, and deterrence has been one focal point of the debate.  There are other reasons to support or oppose the death penalty, but in this project we'll focus on deterrence.\n",
    "\n",
    "The key question about deterrence is:\n",
    "\n",
    "> Through our exploration, does instituting a death penalty for murder actually reduce the number of murders?\n",
    "\n",
    "You might have a strong intuition in one direction, but the evidence turns out to be surprisingly complex.  Different sides have variously argued that the death penalty has no deterrent effect and that each execution prevents 8 murders, all using statistical arguments!  We'll try to come to our own conclusion.\n",
    "\n",
    "Here is a road map for this project:\n",
    "\n",
    "1. In section 1, we investigate the main dataset we'll be using.\n",
    "2. In section 2, we see how to test null hypotheses such as this: \"For this set of U.S. states, the murder rate was equally likely to go up or down each year.\"\n",
    "3. In section 3, we apply a similar test to see whether U.S. states that suddenly ended or reinstituted the death penalty were more likely to see murder rates increase than decrease.\n",
    "4. In section 4, we run some more tests to further claims we had been developing in previous sections. \n",
    "5. In section 5, we try to answer our question about deterrence using a visualization rather than a formal hypothesis test.\n",
    "\n",
    "#### The data\n",
    "\n",
    "The main data source for this project comes from a [paper](http://cjlf.org/deathpenalty/DezRubShepDeterFinal.pdf) by three researchers, Dezhbakhsh, Rubin, and Shepherd.  The dataset contains rates of various violent crimes for every year 1960-2003 (44 years) in every US state.  The researchers compiled the data from the FBI's Uniform Crime Reports.\n",
    "\n",
    "Since crimes are committed by people, not states, we need to account for the number of people in each state when we're looking at state-level data.  Murder rates are calculated as follows:\n",
    "\n",
    "$$\\text{murder rate for state X in year Y} = \\frac{\\text{number of murders in state X in year Y}}{\\text{population in state X in year Y}}*100000$$\n",
    "\n",
    "(Murder is rare, so we multiply by 100,000 just to avoid dealing with tiny numbers.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "    <thead>\n",
       "        <tr>\n",
       "            <th>State</th> <th>Year</th> <th>Population</th> <th>Murder Rate</th>\n",
       "        </tr>\n",
       "    </thead>\n",
       "    <tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1960</td> <td>226,167   </td> <td>10.2       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1961</td> <td>234,000   </td> <td>11.5       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1962</td> <td>246,000   </td> <td>4.5        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1963</td> <td>248,000   </td> <td>6.5        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1964</td> <td>250,000   </td> <td>10.4       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1965</td> <td>253,000   </td> <td>6.3        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1966</td> <td>272,000   </td> <td>12.9       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1967</td> <td>272,000   </td> <td>9.6        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1968</td> <td>277,000   </td> <td>10.5       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska</td> <td>1969</td> <td>282,000   </td> <td>10.6       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "</table>\n",
       "<p>... (2190 rows omitted)</p"
      ],
      "text/plain": [
       "State  | Year | Population | Murder Rate\n",
       "Alaska | 1960 | 226,167    | 10.2\n",
       "Alaska | 1961 | 234,000    | 11.5\n",
       "Alaska | 1962 | 246,000    | 4.5\n",
       "Alaska | 1963 | 248,000    | 6.5\n",
       "Alaska | 1964 | 250,000    | 10.4\n",
       "Alaska | 1965 | 253,000    | 6.3\n",
       "Alaska | 1966 | 272,000    | 12.9\n",
       "Alaska | 1967 | 272,000    | 9.6\n",
       "Alaska | 1968 | 277,000    | 10.5\n",
       "Alaska | 1969 | 282,000    | 10.6\n",
       "... (2190 rows omitted)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "murder_rates = Table.read_table('crime_rates.csv').select('State', 'Year', 'Population', 'Murder Rate')\n",
    "murder_rates.set_format(\"Population\", NumberFormatter)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Murder rates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So far, this looks like a dataset that lends itself to an observational study.  In fact, the murder rates dataset isn't even enough to demonstrate an *association* between the existence of the death penalty in a state in a year and the murder rate in that state and year!\n",
    "\n",
    "**Question 1.1.** What additional information will we need before we can check for that association? Assign `extra_info` to a Python list containing the number(s) for all of the additional facts below that we *require* in order to check for association.\n",
    "\n",
    "1) What year(s) the death penalty was introduced in each state (if any).\n",
    "\n",
    "2) Day to day data about when murders occurred.\n",
    "\n",
    "3) What year(s) the death penalty was abolished in each state (if any).\n",
    "\n",
    "4) Rates of other crimes in each state."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "extra_info = [1, 3]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/lOvWJJ\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q1_1\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Murder rates vary over time, and different states exhibit different trends. The rates in some states change dramatically from year to year, while others are quite stable. Let's plot a couple, just to see the variety.\n",
    "\n",
    "**Question 1.2.** Draw a line plot with years on the horizontal axis and murder rates on the \n",
    "vertical axis. Include two lines: one for Alaska murder rates and one for Minnesota murder rates."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAqYAAAEfCAYAAACTVgS/AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAIABJREFUeJzs3XlYU3f2P/B32HfCGnZQQJRF3MEN1zIuVey07rV1+9bqtHVsXVp/jo7VUTtOq7a2ttNtrEvrUpe22qqtKBa1KCLghiCL7GDYwxZIfn/QhNx7kxBIIEHO63l8Hu/NvTc3F8TD5/M55/AqKiqkIIQQQgghRM+M9H0DhBBCCCGEABSYEkIIIYQQA0GBKSGEEEIIMQgUmBJCCCGEEINAgSkhhBBCCDEIFJgSQgghhBCDQIEpIYQQQggxCBSYEkIIIYQQg9CjAtP09HR930K3Rc9OO/T8Oo6enXbo+RFCupMeFZgSQgghhBDDRYEpIYQQQggxCBSYEkIIIYQQg0CBKSGEEEIIMQgm+r4BQgghpKuIRCI0NTXp+zYI6bFMTExgbW2t+vUuvBdCSDtJJBKkpmXD3tYKfl5u+r4dQrq1hoYGAIC9vb2e74SQnkskEqGhoQHm5uZKX6fAlBADtvO/x3HrTgYAYNm8KRg/YoCe74iQ7qu+vh52dnb6vg1CejQrKytUVVWpDExpjSkhBqqgWCgPSgHgbGyCHu+GkKcDj8fT9y0Q0qO19W+QAlNCDFROfglju7C0DBKJRE93QwghhHQ+CkwJMVC5hczAtKlJAmF5lZ7uhhBCCOl8FJgSYqDyCp9w9hWWluvhTgghPdXw4cOxfft2fd9Gp+Hz+Th9+nS3e6+cnBzw+XwkJSXp5HqGhAJTQgxUbmEpZ18xBaaE9CjLly8Hn8/Ha6+9xnlt06ZN4PP5mD17th7uTH+mTp2KNWvW6ORaaWlpmDRpkk6udfv2bTg6OuIvf/mLTq7XU1FgSogBahSLUaQkCC16QoEpIT2Nl5cXTp06BZFIJN/X1NSE7777Dl5eXlpfv7GxUetrqCKRSNDc3KzRsWKxuNPuQxWBQKAyO7y9Dhw4gCVLluD+/ftIS0vTyTV7IgpMCTFABcVlkEiknP1FpWV6uBtCiD6FhISgd+/eOHnypHzfuXPnYG5ujlGjRjGOXb58OWcEdfv27Rg+fDjnmN27dyM4OBjBwcEAgNLSUsydOxdubm4IDQ3FgQMHOPdSWVmJlStXIiAgAF5eXpgyZQpjOvnQoUPw9PTE+fPnMXz4cLi4uCgN0mRT0cePH8e0adPg5uaGr7/+GmVlZViyZAmCg4Ph5uaGyMhIHDx4kHHv8fHx+Pzzz8Hn88Hn85GTkwMAePDgAWbNmgUvLy8EBARgyZIlKC4uVvtsFafXZfd0+vRpzJgxA+7u7oiIiEBsbKzaawBAXV0djh07hoULF2L69OlKnx3bP//5TwwZMgRubm4ICwvDxo0bUV9fL389Ly8Pc+fOhZ+fH9zd3TF06FB8//33Sq8lkUiwevVq9O/fH48ePQIA7N27FyNGjICHhwf69euH119/HRUVFW3el75RHVNCDFBeEXd9KQCUPDH8HyqEdCezX9vWpe93ZO/6Dp23YMECHDp0CC+++CIA4ODBg5g/fz6ys7M7dL34+HjY2dnh+PHjkEpbfglesWIFcnNzcerUKVhaWmL9+vV4/Pix/BypVIrZs2fDzs4OR44cgYODAw4fPozp06fjxo0bcHNraQJSX1+PnTt3YteuXXB2doZAIFB5H5s3b8bWrVvx0UcfwdTUFPX19QgPD8fKlSthZ2eHS5cuYdWqVfD29saYMWOwY8cOPHr0CIGBgdi4cSMAwNnZGUVFRZgyZQoWLFiALVu2QCwWY8uWLZg3bx4uXLgAIyPNx+G2bt2Kd999F++//z527tyJxYsXIzU1FTY2NirPOX36NLy9vRESEoLZs2dj0aJF2LRpE0xNTVWeY2Vlhb1798Ld3R1paWl48803YWZmhg0bNgAA3nrrLTQ0NODHH3+Era0tMjIylF5HLBbj1Vdfxb1793Du3Dm4u7sDAIyMjLB9+3b4+fkhNzcXa9euxdq1a/Hf//5X42ehDxp9peLj4zFnzhz069cPfD4fhw4dYrwu+62F/Wf16tUqryn7zYT959dff9XuExHyFMhTsr4UAIpKy+X/iRBCeo6ZM2ciKSkJjx49QnFxMX777TfMmzevw9czNzfH3r17ERwcjJCQEGRkZODChQvYvXs3IiMjER4ejn379qGurk5+TlxcHFJTU7F//34MHjwYvXv3xoYNG+Dr64sjR47Ij2tubsbOnTsRGRmJgIAA2NraqryPV155BTExMfDz84Onpyc8PDzwxhtvoH///vDz88PChQsxbdo0HD9+HEBL1y5TU1NYWVlBIBBAIBDA2NgYX375JUJDQ7F582YEBQUhNDQUn332GRITE9udILRixQpMnjwZ/v7+2LhxI8rLy5Gamqr2nAMHDmDOnDkAgFGjRsHS0hJnz55Ve87atWsRGRkJX19fREdH480332SMiObm5iIyMhJhYWHw8/PDxIkTMXHiRMY1RCIR5syZg5ycHJw9e1YelMo+x5gxY+Dr64tRo0bh3XffxalTpwy+7KBGI6YikQjBwcGYO3cuXn31Vc7r7GH6pKQkzJkzBzNmzGjz2t9//z1CQ0Pl2w4ODprcEiFPNWWJTwDQKG5CeWUNHPmqf9ATQp4+fD4fzz77LA4ePAh7e3uMGjUK3t7eHb5ev379GGsr09LSYGRkhMGDB8v3+fj4MAKd5ORk1NbWIiAggHGt+vp6ZGVlybdNTEwQFham0X0MHDiQsd3c3Ixdu3bhxIkTKCwsRGNjIxobGzlLFtiSk5Nx9epVeHp6cl7LyspifK62hISEyP8u+/ylpcp/JgNAZmYmrl+/ji+++AJASwH5WbNm4cCBA4iJiVF53unTp7Fv3z5kZmZCJBKhubmZsR731VdfxZtvvonffvsNY8aMwbPPPosBA5jd/5YtWwaBQIAff/yR03/+8uXL2LVrFx4+fIiqqio0NzejsbERxcXFjK+rodEoMI2OjkZ0dDSAlgicjT1Mf/bsWQQEBLT5jQQAjo6Oaof5CemJlJWKkikqLaPAlJAe6MUXX8Ty5cthbW2N9euVLwkwMjLizKo0NTVxjmMHMTLquvJIJBK4urri559/5rymOCpqbm4OY2NjlddRdx8fffQR9u7dix07diA4OBg2NjZ499131QaGsnuLjo7G1q1bOa+5uLhodC8yitPvsuehbqbqm2++QXNzM2OQTXZ8Xl6e0gS1GzduYPHixVi3bh22bdsGe3t7nD17Fv/4xz/kx7z00kuYMGECLly4gEuXLiE6OhqrVq3CO++8Iz8mOjoa3333Hf744w+MHz9evv/x48eYPXs2XnrpJaxfvx6Ojo5ITk7GkiVLOjXZTRd0vsa0pqYGJ06cwLp16zQ6fsGCBaivr4e/vz9WrFih9rcLQnqChkYxitVk3xc/KUdwoG8X3hEhT6+OrvnUhzFjxsDU1BRCoRBTp05VeoyzszNn2rmtaWgA6NOnDyQSCRITExEREQGgZSq5sLBQfkx4eDhKSkpgZGQEPz+/jn8QNa5du4ZJkybJp8WlUikyMjJgb28vP8bMzIyT6R8eHo6TJ0/C29tb7bpOXWtqasK3336LTZs2ccpELVu2DIcOHVIaD12/fh3u7u5Yu3atfF9ubi7nOE9PTyxcuBALFy7E7t278emnnzIC05deegnh4eGYP38+Dh8+jHHjxgFomblubGzE9u3b5b8k/PLLLzr5zJ1N51n5x48fR2NjI+bOnav2OBsbG2zZsgVff/01jh07hqioKCxatIixToWQnqigWAh1y0gLS6hkFCE9EY/HQ3x8PJKTk1WWOIqKikJKSgoOHDiAzMxM7NmzB9evX2/z2oGBgZg4cSJWrVqFhIQEpKSkYMWKFbC0tJQfM3bsWERGRsoTirKzs5GQkIBt27bh6tWrOvmMAQEBiIuLw7Vr1/Dw4UOsWbOGkYAFtCwxSExMRE5ODoRCISQSCZYuXYqqqiosWrQIN2/eRHZ2Ni5duoSVK1eiurpaJ/emzLlz5yAUCvHyyy/LKxzI/jz//PM4dOiQ0tHWgIAAFBYW4ujRo8jOzsaXX37Jybhft24dfv31V2RnZyMlJQW//vorgoKCONdauHAhtm3bhvnz58srCPj7+0MikeCTTz5BdnY2jh8/jk8//bRzHoKO6XzEdP/+/ZgyZQqcnZ3VHufk5ITXX39dvj1w4ECUlZVhz549aosFp6ena3V/2p7fk9Gz046mz+9GagajXiGPB0agevdBOtKDta9d2J3Q9552DPn5BQYG6vsWuhV1iUQAMGHCBKxbtw5bt25FXV0dZs6ciaVLlyqdfmf75JNP8MYbb2D69OlwcnLCunXr8ORJ67IiHo+Ho0ePYuvWrVi5ciVKS0vh6uqKiIiINgejNLVmzRrk5ORg5syZsLCwwLx58zBz5kw8ePBAfszrr7+O5cuXIzIyEnV1dUhOToavry/OnTuHzZs34/nnn0dDQwO8vLwwbtw4ndUpVebAgQMYPXo0HB0dOa/NmDED//znPxEbG8uYZgeAyZMn44033sA777yD+vp6jBs3DuvXr8dbb70lP0YikWDt2rXIz8+HjY0NxowZo3SpAgAsWrQIUqkU8+fPx6FDhzBu3Djs2LEDe/bswb/+9S8MGzYMW7ZswaJFi3T7ADoBr6Kiol0pvp6envj3v/+N+fPnc15LSUlBVFQUTp48KR9Obo/Dhw/jzTffRFFRUbvP1UR6ejr9EOwgenbaac/zO3T6In640DrC0S/AG/czWqd4/LwEeO/tJTq/R0NF33vaoefXqrKykjElTAjRD3X/FnU6lb9//374+vpi7NixHTo/NTWVEqFIj8dOfBrSvw9ju6i0jEpGEUIIeSppNJVfU1ODzMxMAC1Dy3l5eUhJSYGDg4O8XEVtbS2OHTuGN954Q2lW3+bNm5GYmIgffvgBQMvoqKmpKfr37w8jIyP88ssv+OKLL/DPf/5TRx+NkO6JXSoqtI8vLMxNUd/Q0q6vvkGMyupa8O2UZ9USQggh3ZVGgWlSUhKmTZsm396+fTu2b9+OuXPnYt++fQCAEydOQCQSKZ3iB4CioiJGnTMA+M9//oPc3FwYGxvD398fe/fuVbu+lJCnXX1DI0qFlfJtHg/wEDjBzcUR2XmtrfWKn5RTYEoIIeSpo1FgOnr06Db7q7744ovyVmnKyAJYmXnz5mnVtYKQpxF7Gt/d1RFmpqZwc3FgBKZFJWUI6t2zEqAIIYQ8/XReLooQ0nF5RcxpfC+3luoWAmdmR7QiNXVOCSGEkO6KAlNCDMjjAlZg6t7SscTNhRmYFpdSYEoIIeTpQ4EpIQaEPZVPI6aEEEJ6EgpMCTEg7Kl8bw8aMSWEENJzUGBKiIEQ1dZDWN7aOs/Y2Agerk4AAEe+LcxMW3MVa2rrUS2q6/J7JIQQQjoTBaaEGIi8IlZGvosDTEyMAbS0AhQ48xmv06gpIaSzDR8+HNu3b9f3bXQaPp+P06dPd+l7bt++HcOHD+/S9+xOKDAlxEDkFSqfxpcRuNA6U0J6muXLl4PP5+O1117jvLZp0ybw+fweV/976tSpWLNmjU6ulZaWhkmTJml1jfZ+jV5//XWcOXNGq/fUh0OHDsHT07PT34cCU0IMBDfxiRmYurk4MrZpxJSQnsHLywunTp2CSCSS72tqasJ3330HLy/t6xk3NjZqfQ1VJBIJmpubNTpWLBZ32n2oIhAIYG5urvV12vM1srGxgaOjI/sS5E8UmBJiIHLZNUzdnRnbbqzM/MLSsk6/J0KI/oWEhKB37944efKkfN+5c+dgbm6OUaNGMY5dvnw5ZwSVPXUsO2b37t0IDg5GcHAwAKC0tBRz586Fm5sbQkNDceDAAc69VFZWYuXKlQgICICXlxemTJmCpKQk+euyUbXz589j+PDhcHFxQVpaGuc6OTk54PP5OH78OKZNmwY3Nzd8/fXXKCsrw5IlSxAcHAw3NzdERkbi4MGDjHuPj4/H559/Dj6fDz6fj5ycHADAgwcPMGvWLHh5eSEgIABLlixBcXEx570VKU7ly+7p9OnTmDFjBtzd3REREYHY2Fi11wDa9zVS9fXYt28f+vXrB19fX6xYsQK1tbXyY6ZOnYq33noL7777Lnr37o2AgABs2LABEolEfkxjYyM2bdqE4OBguLu7Y9y4cfjtt9/kr4vFYqxduxZ9+/aFq6srQkJCGG3gKyoq8Oqrr8LX1xdubm6IiYnB/fv3AQBXrlzB3/72N4hEIvlzly3xOHLkCMaNGyd/7i+//DIKCgrafGaqaNT5iRDS+XJV1DCVYU/l04gpIdqzyl3bpe9X6/3vDp23YMECHDp0SN5h8eDBg5g/fz6ys7M7dL34+HjY2dnh+PHjkEqlAIAVK1YgNzcXp06dgqWlJdavX4/Hjx/Lz5FKpZg9ezbs7Oxw5MgRODg44PDhw5g+fTpu3LgBNzc3AEB9fT127tyJXbt2wdnZGQKBQOV9bN68GVu3bsVHH30EU1NT1NfXIzw8HCtXroSdnR0uXbqEVatWwdvbG2PGjMGOHTvw6NEjBAYGYuPGjQAAZ2dnFBUVYcqUKViwYAG2bNkCsViMLVu2YN68ebhw4QKMjDQfh9u6dSveffddvP/++9i5cycWL16M1NRU2NjYqD1Pm6/RtWvXIBAIcOrUKeTn52PhwoUICAjAm2++KT/m2LFjWLZsGc6fP4/U1FQsXboUAwYMwAsvvAAA+Nvf/oasrCx8/vnn8l8O5syZg4sXLyIsLAyffvopzpw5gy+//BI+Pj4oKChAenq6/PrLly9HRkYGDh8+DD6fjy1btuCFF17AzZs3ERERge3bt2PLli3yX0SsrVvaYjc2NuKdd95Bnz59IBQKsWnTJixZsgQ///yzxs9cEY2YEmIAqkV1qKhqnQIyMTHijJC6swNTWmNKSI8xc+ZMJCUl4dGjRyguLsZvv/2mVVtvc3Nz7N27F8HBwQgJCUFGRgYuXLiA3bt3IzIyEuHh4di3bx/q6lqrf8TFxSE1NRX79+/H4MGD0bt3b2zYsAG+vr44cuSI/Ljm5mbs3LkTkZGRCAgIgK2trcr7eOWVVxATEwM/Pz94enrCw8MDb7zxBvr37w8/Pz8sXLgQ06ZNw/HjxwEA9vb2MDU1hZWVFQQCAQQCAYyNjfHll18iNDQUmzdvRlBQEEJDQ/HZZ58hMTGRMaKriRUrVmDy5Mnw9/fHxo0bUV5ejtTU1DbP0+ZrZGtri127diEoKAjjx4/HjBkzcPnyZcYxQUFB+H//7/8hICAAzz33HEaPHi0/JisrC8ePH8fXX3+NkSNHws/PD6+88gqeeeYZ/O9//wMA5Obmwt/fHyNGjIC3tzciIiLkQfSjR4/w888/Y/fu3Rg5ciRCQkLw2Wefobq6GseOHYOZmRns7OxaEnH/fO6yQH3BggWIjo6Gn58fBg8ejA8++ADXrl1Dfn6+po+cgUZMCTEA7MQnD4GTPCNfxsnBDiYmRmhqapm6qayuhai2HtZWFl12n4QQ/eDz+Xj22Wdx8OBB2NvbY9SoUfD29u7w9fr168dYW5mWlgYjIyMMHjxYvs/Hxwfu7u7y7eTkZNTW1iIgIIBxrfr6emRlZcm3TUxMEBYWptF9DBw4kLHd3NyMXbt24cSJEygsLERjYyMaGxs50+FsycnJuHr1qtLknKysLMbnaktISIj877LPX1paqupwOW2+RkFBQTA2bv2Z7+bmhps3b6q8L9kxsvtKTk6GVCpFZGQk45iGhgZERUUBAObNm4fnnnsOgwcPxvjx4/HMM8/gmWeegZGRkfzrP2zYMPm59vb2CA4OxoMHD9Te++3bt/Hee+8hNTUVFRUV8hH4vLy8DiVLUWBKiAHIZWfksxKfAMDIyAiuTnwUFLeuLS0RVqCXlVun3x8hRP9efPFFLF++HNbW1li/fr3SY4yMjOSBgUxTUxPnONk0LBuPx1P5/hKJBK6urkqnaBVHRc3NzRlBljrs+/joo4+wd+9e7NixA8HBwbCxscG7777bZmAokUgQHR2NrVu3cl5zceH+PFXH1NRU/nfZ82A/U1U0+Rq19Z6y92W/p7pjJBIJeDweLl68yDnOwqJl8GLAgAFISUnBxYsXcfnyZSxfvhyhoaE4deqU2ntT9z0hEonw/PPPY+zYsfjss8/g4uICoVCIyZMndzipjgJTQgwAOyOfXSpKRuDswAhMi0rL0cubAlNCOqqjaz71YcyYMTA1NYVQKMTUqVOVHuPs7MyZdtZkGrpPnz6QSCRITExEREQEgJap38LCQvkx4eHhKCkpgZGREfz8/Dr+QdS4du0aJk2ahDlz5gBoCQgzMjJgb28vP8bMzIyT6R8eHo6TJ0/C29ubE5h1JU2+Rp2hf//+kEqlKC4ulo+QKmNra4uYmBjExMRg3rx5mDhxIjIzMxEUFASJRIKEhASMHDkSAFBVVYV79+7JlyMoe+7p6ekQCoX4xz/+If+e+OGHH7T6LLTGlBADwJ7K93JzVnocuzVpEWXmE9Jj8Hg8xMfHIzk5WWWJo6ioKKSkpODAgQPIzMzEnj17cP369TavHRgYiIkTJ2LVqlVISEhASkoKVqxYAUtLS/kxY8eORWRkpDyhKDs7GwkJCdi2bRuuXr2qk88YEBCAuLg4XLt2DQ8fPsSaNWsYCVhAyxKDxMRE5OTkQCgUQiKRYOnSpaiqqsKiRYtw8+ZNZGdn49KlS1i5ciWqq6tVvJvuafI16gwBAQGYNWsWVqxYgdOnTyM7OxtJSUn46KOP5IHi3r17cfz4caSlpSEzMxPHjh2DnZ0dPDw84O/vjylTpmDVqlW4evUq7t69i1deeQW2traYOXMmgJbnXl9fj9jYWAiFQtTW1sLLywvm5ub4/PPPkZ2djXPnzmHbtm1afRYKTAkxAOyuT+yMfBmBMzsBqqLT7okQYnhsbW1hZ2en8vUJEyZg3bp12Lp1K8aOHYvHjx9j6dKlGl37k08+gY+PD6ZPn465c+di5syZ8PHxkb/O4/Fw9OhRjB49GitXrsTQoUOxaNEiZGRkMNaiamPNmjUYNGgQZs6ciSlTpsDKykoeGMm8/vrrMDMzQ2RkJPz9/ZGbmwt3d3ecO3cORkZGeP755xEZGYnVq1fDzMysSwNEoO2vUWf5+OOPMX/+fGzcuBFDhw7F7NmzER8fL/8a2tra4sMPP8SECRMwZswYpKam4tixY7CysgLQ8vUfNGgQ5s6diwkTJqCurg7Hjx+X/3ISERGBxYsXY8mSJfD398eePXvg7OyMffv24cyZM4iIiMB7772Hf/3rX1p9Dl5FRYVmCyeeAunp6QgMDNT3bXRL9Oy0o+75VVaL8Mo7e+TbZqYm2P/+aqXlTW7fe4Ttn7RmvwYH+mDTyhd1f8MGhL73tEPPr1VlZSVjSpgQoh/q/i3SiCkhesZOfPIQOKmsucceMS0soal8QgghTw8KTAnRM00TnwDAxdEeRkatGZLllTWob+i8doKEEEJIV9IoMI2Pj8ecOXPQr18/8Pl8HDp0iPH68uXL5S2qZH8mTpzY5nV///13jBkzBgKBAOHh4fjqq6869ikI6cbYiU/eKhKfAMDExBgujszpD1pnSggh5GmhUWAqEokQHByMHTt2MDL0FI0dOxZpaWnyP8eOHVN7zezsbMyaNQvDhg1DXFwc3nzzTaxdu1bes5aQnoJTw1TNiCmgpDUpdYAihBDylNCojml0dDSio6MBtLTqUsbc3FxtP1y2r7/+Gm5ubti5cyeAlq4HN2/exN69exETE6PxdQjpzqRSKWcqX1VGvoybswNS0NplpaiUAlNCCCFPB52tMb127RoCAgIwePBgvPHGG212aUhISMD48eMZ+yZMmICkpCSIxWJd3RYhBq2iqgY1tfXybXMzE85UPZubqyNjm0ZMCdGcph18CCGdo61/gzoJTCdOnIhPP/0Up0+fxtatW5GYmIjp06ejoaFB5TklJSWcNmEuLi5oamqCUCjUxW0RYvBylYyWqmv/BgACJz5ju4gy8wnRiIWFBWpra/V9G4T0aLW1tfI2qcropCXp888/L/97SEgIBgwYgLCwMJw7dw7Tp0/XxVvIpaen6/X8noyenXaUPb+ExLsQiUTybXMTaZvPuU5UyTjnYWbOU/+1edo/X2cz5OfXlTVWzc3N0dTUhMrKyi57T0IIk4mJidqmBzoJTNnc3d3h4eGBzMxMlce4urpypvtLS0thYmICJycnledp80OMCk13HD077ah6fhdvPIS1tbV8e1D/4Dafs6+fGDaHL0A2GyJuBnz9/GCmx/7QnYm+97RDz49J8d8bIcTwdEodU6FQiMLCQrXJUMOGDUNsbCxjX2xsLAYOHAjTp/Q/WELYODVM20h8AgAzU1M48m3l21IpUCKkESBCCCHdn0aBaU1NDVJSUpCSkgKJRIK8vDykpKQgNzcXNTU12LBhAxISEpCTk4MrV65gzpw5cHFxwbPPPiu/xrJly7Bs2TL59qJFi1BYWIi3334baWlp+Oabb3D48GG89tpruv+UhBggqVSKvCJWqSgNAlMAcHdhJUBRZj4hhJCngEaBaVJSEqKiohAVFYW6ujps374dUVFR2LZtG4yNjXHv3j3MmzcPQ4YMwfLlyxEQEIDz58/D1rZ1VCcvLw95eXnybT8/Pxw9ehRXr17F6NGj8Z///AfvvfcelYoiPUZFVQ1q61q7NllamDFGQtVh1zItosx8QgghTwGN1piOHj0aFRWqu8ucOHGizWucOXOGs2/UqFGIi4vT5BYIeeo8KatibAucHdrMyFc8VhGNmBJCCHkadMoaU0JI256UMwNTZ0c7jc91Y4+YUmBKCCHkKUCBKSF68qScmbDkxG9PYEpF9gkhhDx9KDAlRE+ErBFTJwfNA1OBM7PIfomwAk1NzTq5L0IIIURfKDAlRE/Ya0yd2xGYWpibwcHeRr4tkUg5I7CEEEJId0OBKSF6IqxgrzG1b9f5tM6UEELI04YCU0L0hD3Pn/mSAAAgAElEQVSV354RUwBwdWJO51NgSgghpLujwJQQPWgUi1FZXSvf5vEAvl37WiW6u1KRfUIIIU8XCkwJ0QNheTVj25FvC2Nj43Zdg13LtLC0TOv7IoQQQvSJAlNi8Morq5GWmYdGsVjft6Iz7EQlZ4f2rS8FAA8Bc8T0YVYeZeYTQgjp1jTq/ESIvtx9mINtn3yLpiYJ+gV4Y/3f5sDM1FTft6W1MtaIaXvXlwKAj4crbK0tUS2qAwCIahuQlpmHkD6+OrlHQgghpKvRiCkxaD/8eg1NTRIAwP2MXFy6nqLnO9INdtcnp3Z0fZIxMjLCgBB/xr7EO+la3RchhBCiTxSYEq00NTXjyyO/4M0tn+HomcuQSqU6u7ZUKkVWbhFj34+/Xkdzc/efri4tY0/ltz8wBYDBoYGM7VsUmBJCCOnGKDAlWjl6Jg7nr9xCfrEQ3/8cj5QHWTq7dkVVDSNzHQBKhJW4lvRAZ++hrdzCUnxz4lfsP34BZRXVbZ/wJ3YN0/a0I1UU3q83jI1b/xkXlpSjoFjYoWsRQggh+kaBKemwJ2WVOHspgbEvNU13gWlOfonS/acvXNXpyGxH1Dc04sDJ37Bux5c4czEBZy/dwCcHftT4fCG761MHpvIBwMrSHMEBPox9NJ1PCCGku6LAlHTYdz9dhljMnFbPK3yis+vn5BUr3f84vxRJdzN09j7tIZVKce3Wfaza8hl++u0PNDdL5K+lpmVDVFuv0TXYWflOHcjKlxkUGsDYvnVHP8+GEEII0RYFpqRDsvOK8PuNO5z9eUU6DExVjJgCwKnz13T2PpoqLCnDtk++w+6vTqqcti8sabuWaE1tPRoam+Tb5mYmsLGy6PB9DQ5jrjN98CgXNbV1Hb4eIYQQoi8UmJJ2k0qlOHjqIpTNppcKK1FX36CT93lcoDowTcvMw4NHuTp5n7Y0isU48tNlrN72X6TcV79UoaCk7fWdQiU1THk8XofvT+DsAC83J/m2RCJF8r3MDl+PEEII0RcKTLuB2Gu3sXbHF/jw61PympX6lHw/E6kPslW+rovp/EaxGPmsJJ6g3l6M7VPnr2r9Pm3JzivCW1v/ixO/xMvLVskYGxtxitxrMmLK6frkYKv1fQ5ijZomptI6U0IIId0PBaYGrkRYgc8On0VOXgniE+91STCmjkQiwaHTF9Uek1dUqvX75BcJIZG0Dsm6ONljzrQxjGOS7j5SO92vLalUio+/+RElwkrOayGBPvj3O0sxfWIkY3+hBiOmuuj6xMYuG3X7/iPqAkUIIaTbocDUwKVn5TOmzJPvPdLfzQCIS0jF43xm4MlOvsnVwYhpTj4z8cnX0xX9AnwQ6OfB2H+6EwP1/GIhHhcwP6u9nTVeXxiDf7wxH15uznBzYY+Ylrd5XXZxfRdH7QPTPr08YWttKd8W1TbgYVae1tclhBBCupJGgWl8fDzmzJmDfv36gc/n49ChQ/LXxGIxNm3ahBEjRsDDwwNBQUFYunQpcnPVr/+7cuUK+Hw+58/Dhw+1+0RPmfKqGsZ2XtET1Dc06uVeGhrFOPJTHGPfqKEhiBoWxtiXW6j9iGlOHnMk1NfTFTweDzOiRzD2X711D8VP2g4GO+JGchpjO9DPA7s2LMOoISHyNaHurk6MYwpLhG2WsmKXinLSwVS+8i5QlJ1PCCGke9EoMBWJRAgODsaOHTtgaWnJeK22thbJyclYvXo1Ll++jMOHDyM/Px8vvPACmpqaVFyx1fXr15GWlib/4+/v3+Y5PUl5JTMwlUrVZ6t3prOxCYxsdBMTI8x5dgy83J0Zx+lijSl3xFQAoCUD3Vvh/aTSlm5QneFGCvOXpLGR/WHNyp63t7WCtZW5fLuhsanNQvucdqQd7PrERl2gCCGEdHcaBabR0dHYuHEjYmJiYGTEPMXe3h6nTp3CX//6VwQGBmLw4MHYtWuXPNBsi4uLCwQCgfyPsbFxxz7JU6qikhvkZD4u7PL7qBHV4fQFZommSWOGwsWJD3cXR0b3obKKao3qeaoilUo5wbevpysAgMfjYfozwxmvXfojBRWskWVtCcur8Cin9TnzeNyyTLL74Uznl6pPgOLUMO1g1ye2/n17Mb4OBcVl1AWKEEJIt9Ipa0yrq1uCKT6f3+axY8eORVBQEKZPn464uLg2j+9pKqpEnH2ZrP7xXeF8fDLq6luXENhYWeC5v7RMq5uYGMPDlRmcaVPPtKyiGjUKga2FuSkEzg7y7RGDguHi1LouUyxuxtnYGx1+P2VupjJHSwP9POFgr3zK3UPAnM5XFww2NzdzRsE72vWJzdrKgtMF6paeGhEQQgghHWGi6ws2NjZiw4YNmDRpEjw9PVUe5+bmhg8++ACDBg1CY2Mjjhw5gpiYGJw5cwYjRoxQeV56unbTk9qe39WyHudBJGIGp0mpD5Ce3rfL7qFEWImrt9LQrJAlPzEyGIX5rck1FqY8xn1ev3kbRs0dK211NyOXcS0XvgsyMpgB1qC+3jhxvkC+feLsJYT2doWlhVmH3pPtXOx1xj14udiq/N7hNTcwjk2+8wC93LjBZnp6OiqqRKiubg1MbawtkJOdrZN7BgA3J2tcV7iXX+MSEOTtqOaM7qG7/bs1NIb8/AIDuTMRhJCeS6eBaVNTE1555RVUVlbi22+/VXtsYGAg4wfSsGHD8PjxY3z44YdqA1Ntfoilp6d3ux+CUp4xrK2tGftq6sTw9vGFhblugrC2/Hj5ezRLpPL7cHWyx8LZz8LUtPXbZ2BYMdIftyY9SY3MOvys72SWMD7zgNAgzrV8fP1wPSUTldW18n2ZhZWc5KiOqKmtQ6GwmnEP0yeNhbur8gCvtEqMuMTW//ibYcK5X9n3XlpmHuO6ft4CnX5P2vKdcOHqXfl2cVkN3D29YGNlqeYsw9Yd/90aEnp+hJDuRGdT+U1NTViyZAnu3r2L06dPw9Gx/aM0gwcPRmYmdayRqW9oRG0dNwO/KxOg0rPy8cdt5lrhOdPGMoJSADpNgGInPvn8ub5UkbmZKSaPHcrYdzY2QSetOG/dyWDUUPVyc1IZlAJoV5H9J2W6r2GqyM3FkbpAEUII6bZ0EpiKxWIsWrQId+/exY8//giBQNCh66Smpnb43KcRey2ioq5KgDobm8DY9vd1x4jBwZzjvN1dGNuPtSgZxU18Uv49ET16MGPqvrK6Fv89fLbNck1tucnKxh8aHqT2eHbyU/GTcpXF7dkZ+bpaX6qI3QWKsvMJIYR0FxoFpjU1NUhJSUFKSgokEgny8vKQkpKC3NxcNDU14eWXX8bNmzfxxRdfgMfjobi4GMXFxairax29WrZsGZYtWybf/uSTT/DTTz/h0aNHuH//PjZv3owzZ87g//7v/3T/KbupciUZ+TJdkQBVUVWDP1i1PGc/O0ZpX3eBMx8mJq3fTpVVog61T21oFHNGHH08XJQea21lgSnjmKOmf9xOw6+/J7X7fWUaxWLcZjUxGNq/j9pzLMzN4MhvTYySSqGytmpZReeUilLE6QJ1LxPNzdQFihBCiOHTKDBNSkpCVFQUoqKiUFdXh+3btyMqKgrbtm1Dfn4+zp49i8LCQnmGvezPiRMn5NfIy8tDXl5rsoxYLMbGjRsxcuRITJ48GdevX8fRo0cxffp03X/KbkrdiGlWF4yYxl5LRnNza394D4Ej+vftpfRYY2NjTnZ6bkH7lxvkFpQyOl25uTjA0sJc5fHPRY9Ebx83xr79Jy50eKlDyv0sNDS21t915Nuit497m+exqxKoms7njJjqeCofaGkEoNgFqqa2HmmZ1AWKEEKI4dMo+Wn06NGoqKhQ+bq612TOnDnD2F65ciVWrlypydv3WBXV3FJRMrIOUJ2VACWRSHDh91uMfc+MGqR0tFTGx92F0a40r+gJggN92/W+ylqRqmNqaoKVi2Zg3Y4vUd8gBtBSPurDr0/iX2sWtfv5sIvqD+3fR+1nlnF3dcSdhzny7QJVgWkndH1iMzY2xoAQf1xJuCPfl3gno91fC0IIIaSrdUodU6Ib6joIdXYC1K07GRCWt76/qYkxxkT0V3uOF2udaW5B+9eZarq+VJGbiyOWzpnM2JdXJMT+7y+0672bm5uRmMpcjzk0XP00voy7gNuaVBlhF4yYAtQFihBCSPdEgakBY3czUuzqA3RuAtT5K8zR0kEhvTntONnYCVAdKbLf3hFTmdFDQzEmIoyx7+LVZFxNvKfxe6dl5jHWxdpYWaCfv4+aM1ppMpVf39DIuL6xsRH4dtac43RBWRcoddUCCCGEEENAgakBq2CtMWV39cnqpASootIyJN9nlhgaOUh9ZjrALRnVsl5U8wx5qVSKxwXKW5FqYvGsv3BKN33+3VmViUhs7Gz8gaEBMDHRrEUuu5yUsql89mipI9+W0+JXV6ytLNDP35uxj7pAEUIIMXQUmAJIvp+J1zZ9jNc2fYzUtCx9345cGSsrf1BoAGO7s0ZML7Cy2gP83OHNCjqVETg7wEyhvmm1qI5RAL8tT8oqGXVbrSzN4Oyo+VS3hbkZVi6awagOUFvXiA//d0pl+SYZqVSKBCXrSzXl4mjPqUogUmirCgBC1tIM507IyFfELhv1R9KDTn0/QgghRFs9PjCtqqnFB198j1JhJUqFlfjfsfP6viU59lQ+OzCVJUDpUqNYjMvXUxj7okcN1uhcHo8HTzfmWsu8Is3XmbLXl/p4uGqUeKTIz8sNL86YwNiXkV2II2cut/HexSgVtha/NzM1QXi/3hq/r7GxMQRODox9RaXMUVNhObO4fmeUilLEDqzTMvM490QIIYQYkh4fmP7423V5NjfQkjTT0ChWc0bXYHd9MjExgsDZAW4urcFPZyRAXbv1gLPOcvjgfhqfr00CVEcSn5SZNGYIhvRnjhb+cOE6pz6pohvJzNHS/v16tTuj312gfjqfk5HP79zA1NWJj77+Xox9ipn6hBBCiKHp0YFpZbUI5y7fVLpf39ijpXw7G/B4PPTyZgZrup7Ov3AlkbE9dng4zExNNT5fmwSojiY+sfF4PLw6/1lOKaYPvvgecQmpSs9JSGE2EmjPNL4Me50pOzO/K7o+sY0eGsrYvnLjjtadsQghhJDO0qMD0x9+vc4opi7DDgr1gV0qysHOBgDQ28eDsV+XCVBZuUVIzy5g7Js4cmC7rsFei6qPEVMAsLW2xGsvTYfiSoCGxiZ8/M2P+O+3Z9Eobh0VL35Szqi/yuMBA0OYyyY04eHKKhlVzJ7KZ5eK6vzANHJQP5iatiZwFT+pwMOs/E5/X0IIIaQjemxgWl5ZjfNxiUpfq6gyhBFT5j042MsCU2aXI10GpudZo6X9+/XijAK2hTtiqllmfl19A4pKW7PneTzAW0UrUk0FB/pi7rSxnP2/xd/Ghvf3y8snsYvq9wvwgb1t+8s4KS6zALhT+UJOO9LOqWGqyMbKEkNYSVCqRo0JIYQQfeuxgekPF66jUcwdLQUMY8S0nHUPDvYt09K9vJiBaW5hqU7WxIpq6xF/8y5jX/SoQe2+jrOjPSzMW6f+RbUNGj3Px6yRVXdXR5ibab6EQJWY6BF4df4URrUAAMjJK8E7//4K127d55SJGtKBaXwA8BAwR4uLSsvkQblUKuWsMe2KEVMAiBrGrO967dY9xogxIYQQYih6ZGBaVlGNC/G3VL5uCIEpZyr/zxFTaysLTgJUdh5zbWZHxCWkMpY1ODnYcqoAaILH43HqmbKDTmUec6bxO7a+VJlxwwdg6+qFcHdljmjW1Tdi91cncT8jl7F/WAcDU3tbK1hZtiZM1TeIUf5nLVpRbT3jFyELc1NYWZp36H3aq3/f3rC3tZJvi2obkJhKNU0JIYQYnh4ZmJ46fxViseq6loYxlc8eMbWR/13XCVBSqRQXWJ2eJowYCGNjzYrLs3m5tT8Bipv41PH1pcr4erpi25rFGD6or9rj/LwEcHHid+g9eDwe3FnrTAv+TICqYNVzdXawa3cprI4yMTHGiMHBjH1XbtB0PiGEEMPT4wLTJ2WV+O0qs4B8/369GNuGMGLK7vrEt2sNTHWdAHU3PQf5xa0Z5MbGRhg/IrzD11PWAaot3MQn3Y2YylhZmmPlouewZPZfGMXwFXUkG18RJzP/zwQo9tKMrlhfqog9nX/7XqZBVJ8ghBBCFPW4wPTk+atoapLIt12c7DHjmRGMYwxhxLS8UvkaU0D3CVC//s4cLR0W3ofxfu3FSYAqVB+YKm9FqtsRUxkej4fo0YPx7qqX4OLEDQ6HhrfdelUdTmb+nyOm5ZXM76muKBWlqJe3G7wUmh80N0twNfFel94DIYQQ0haTtg95epRV1uDS9WTGvucnjeQECYYwYlpepXyNKaA6AUpdslBzczNKhJUoLBGioKQMhSVlKCgWorCkjBMEP6NhpydVlNUylUqlKqeui5+UM5oc2FhZwJHf8cBYE/6+HtixbjH2HfwJN1PSAQCRA/vCR8tKAJwR0z8rDbB/2emqxCcZHo+HqGFhOPzDJfm+uIRUTB47tEvvgxBCCFGnRwWmF+KTGaOlAmc+Rg8NQ1Mzc71pZbVIbSDV2ZR1fbK1tpRvyxKgZOWVWjpAFaNPLy/OtSQSCb4+dh6x15PVrquV8XJzQnCgj1b378i3hZWlmfwz1NU3QlhepbLvvbJp/K549jZWllj9fy8g83EhamrrERLoq/X7erC7P/25RKKcFZh2dtcnZUYNDcW3P16CrHpX5uMi5BU9gZebs/oTCSGEkC7SY6byi0rLkJDCzER+ftIomJgYw8LcDJYWrdnUTU0S1NTW6+y907Py8f7nx/H1sXOoqa1r83hVXZ8UcROglE/nf//z7zh/5ZZGQSkATB47VOvgjMfjwduduUZUXQJUZyc+qcPj8eDv64Hwfr1hYtKxZC9Fbi7MwLREWIGmpmbOiKlTF0/lA4CTgx1C+/gx9sX9QUlQhBBCDEePCUxP/BIPiaS10Lu7qwNGDQ2Rb/PtmAXVdZUYUt/QiPc+PYqE5If45XIivlWYSlVFVdcnRb283RnbyjLz76Xn4Ptffm/z/YyNjeAhcMSsqVGY0M5OT6pwEqDUrDPljJh66T7xqatYmJsxliFIJFIUPynnjJh29VS+zGhWEtTvN+9AIpGoOJoQQgjpWj1iKr+wpIzT7eb5yaMZ5ZD4djYoLGntPFRRVaOTKc707HxUi1pHSZPutl0/UlXXJ0X+vszAlJ0AVS2qw0f7f4Bi0yVzMxP09nGHu6sjPARO8HB1grurI1yd+DoZLVTEfnbqMvN12YrUEHi4OjJ+ucgreoKqmlpYWbX+8qOPqXwAiBgQhC+P/CyvWSssr8bd9ByEBfVq40xCCCGk82k0YhofH485c+agX79+4PP5OHToEON1qVSK7du3o2/fvnBzc8PUqVNx//79Nq97+vRpREREwNXVFREREfjxxx879ina8Gt8EiNA8xQ4YSSrriN7xJSdENRR7IBMWF6N2roGteeo6vqkSF0HKKlUin0Hf+SMvK5+ZSb++fcFWDZvKqZNiMTgsEB4CJx0HpQCgJeSBChlRLX1KBVWyreNjHjwdHNSemx34cZKgLqXnsP4/rO3s4apqX5+J7QwN8OwAcxarnEJd7S6ZrWoTqO2s4QQQkhbNApMRSIRgoODsWPHDlhaWnJe37NnDz7++GO89957uHjxIlxcXPDcc8+hurpaydVaJCQkYPHixZg5cyauXLmCmTNnYuHChbh582bHP40K86aPxd9emgZnh5YA7/nJo2BkxPzofNZ0ua4y85UFZAUKNUOVKa9UnZEvo6wDlGyt5i+Xb3I6+8Q8Mxz9+3bdqBg7uz2vsFRp8MIuE+UpcIKZqfatSPXJQ8AMrFMfZDO29TWNL8OuaZpw+wHqGxpVHK1ao1iMrXsPY+m6XVi74wtU1dS2fRIhhBCihkaBaXR0NDZu3IiYmBhOQCeVSrFv3z78/e9/R0xMDIKDg7Fv3z7U1NTg+PHjKq+5b98+jB49GqtXr0ZQUBBWr16NUaNGYd++fdp9IiWMjY0RNSwMb7/yHFYumoHhg/pxjuEGprpZY5pXyA1M89vohMStYcoNTAHlCVDZeUU4eOo3xv5APw/Mmhqlye3qjL2tNaOSQENjE0qEFZzj2O1UO6Owfldzd2G2Ps1n/SKi78A0tI8v43uqvkGMhOS0dl/n59gb8qD7cX4pp3sYIYQQ0l5aJz/l5OSguLgY48ePl++ztLTEiBEj8Mcff6g878aNG4xzAGDChAlqz9GWsbERRgwO5gTXQOckP0mlUuQVcddWsgMVNnVdnxSxE6Dupedg91cnGSWxrCzN8MaiGZ0yXa8Oj8fjJECxg/Q/bj/A0TOXGft8uvn6UgCctqRs+g5MjYyMMHpYKGMfew12W5qamvFLXCJjX2audq1xCSGEEK0XuhUXt4x4ubgwp25dXFxQWKj6P6ri4mKl55SUlKg4o0V6enoH71T9+VUVZRCJWoPRzOzHWr9XRZUIJaVlnP0p99IwNJhbc1Qm+3E+414qy58gPZ2bOW0KMeO4i/HcEasXooegsqwUlWVttwVtS3ufh7mxlHF/N5JSYGcBNDU348eLNxF3g7sO2cyoSevnrm/NzRLU19WiWcJcuiB7Fg11Ir1/Rm9nG8bX5o9bd3Hj1m3wba3VnNXq1t1M5OYzE+7uP8zstM+l7+fV3Rny8wsMDNT3LRBCDEi3y8rX5odYenq6yvNNLW1h/dM1+baxibnWPzBv33sEa2vuf/T1Yqnaa0t4RozzBoaHwc7GinOch6c39p9WXQ5qwsgBmDk9up13rZy6Z6fKoMJK3E7Lk283wRR8Rxfs/vokMrILOc9m9LBQTBo/Sm+NDXSpt58XCopbfykRiUTyzxsWEqT3/4wDA4EzV1IZSynySkUYOmhAm+dKpVL87/TvnK9fvViK3r17M6pd6EJHvvdIK3p+hJDuROupfIGgZeq1tJQ5IldaWgpXV9XrBQUCQbvP6Uz2rJEiXSQ/qcpELyotR6NYrPS1tro+KWInQCnycnPCy88/08471i0vN+aIeOqDLLz93lfIyGaOpJuYGGHhC8/gbwumPRVBKcBNgFLk7KC8A1ZXi4pgJkGduZgAkQaNJdKz8zlfQ6BlpLhEocICIYQQ0l5aB6a+vr4QCASIjY2V76uvr8e1a9cQERGh8ryhQ4cyzgGA2NhYted0JjsbKyjGRFU1dWhq0qxbkirKEp+Algx6xZqpijTp+qSInQAFAGamJli5+DmYm+k3u529xrSiSsTpqOXiZI/Nf39JJx2nDAm7A5QiJwdu+S99iBoWxuh4Vi2qw6kLV9s872xsgsrXCkvUr58mhBBC1NEoMK2pqUFKSgpSUlIgkUiQl5eHlJQU5ObmgsfjYfny5dizZw9++OEH3Lt3DytWrIC1tTVeeOEF+TWmT5+OzZs3y7dfffVVxMXFYdeuXXj48CE++OADXLlyBcuXL9f9p9SAsbExZ7pc2wSo3ELV62VVlYziZOSrSHySYSdAAcDLz0+Ej4f+s9vtba1hb8tdgiAzKDQA29cuRoCfRxfeVdfwcFUemJqYGKlMZutqttaWiHlmOGPfz5duoFRJ9QSZUmEFric9UPl6YanyX7gIIYQQTWgUmCYlJSEqKgpRUVGoq6vD9u3bERUVhW3btgEAVq5cieXLl2PNmjUYN24cioqKcOLECdjato4MZWVloaioNVkiIiICX331FQ4fPoyRI0fiu+++w1dffYUhQ4bo+CNqzt5Od9P5UqlU5YgpoLpklKalomQGBPsztiMH9tVZW1Fd8GYV2gdaiujPnzEea5fNVLlMobtzVxGYOtrbGtTI8JRxwxgtVMXiZhw5E6fy+F/iEqGuln5bNXoJIYQQdTRKfho9ejQqKlSPovB4PLzzzjt45513VB6TmsotRxMTE4OYmBhNbqFL8O1s8Di/dd2rNiOmT8oqUd+gfB0poCYwVTKVr46vpyuWzZuCi1dvw89LgBefm2BQgU+QvxfuPMyRbzvY2+Dvi59DX39vPd5V51O1xtTZ0TDWl8qYm5li9rNR2HfwjHzflYQ7eHb8MPixuovVNzQi9tptxr4h/QNxM6U147uohFuFghBCCNGU1mtMnybsaXNtiuyzE5/MWC0o84pVjZgyuz4pjmapMn7EAGxdvRBL50yGhblZm8d3panjIhAS6AMzUxMMH9QX77295KkPSoGWZQyK6zdlnPRcw1SZqGFh8PFkjmwfPHmR06nr0vUUiGpb2+na2Vhi5pTRjGMKKDAlhBCiBQpMFeiyLenjAmbFgYEhzCn3wpIySCTc2qTsqXx24f/uxtrKAhtXvoj//ect/H3xXznVD55WPB5P6aipvovrK2NkZIT5McxmF6lp2Ui+nynflkql+PnSDcYxE0cNgpebC4yMWkfoyyqqO9TetLtqbm7WOkmSEEJIKwpMFXDXmOpuxLSvvzcjEUgsblbaopPd9cnB3jAyuLWl69qW3QG7NSlgmIEpAIT3642wvn6MfYdOXZT/8nTrTgaKFBKbTEyMED16EExMjOHqxGecV6SkqcTTKC0zD69t+gSL176PM2oqFRBCCNEcBaYK2F1vtBkxzStkjph6e7jA041ZPim/iJsowk1+ejoC055IWWtSQ1tjKsPj8fDijPGMkmmPC0px+Y8UANwSUcMHBcu/N9kjw4qNBQyZqLYeu748gbU7vsDvN++2+/z9319AWUU1GhqbcODErz0mICeEkM5EgakC7lR+x0ZMlWXke7u7cP4DV1aAv6L66ZrK78mUZeZrsmZYX/y83DBqaChj35Gf4vAwK4+RwAYAz44fJv87e2S4sJusM/38u7O4nvQAOXkl2HfwR7VlstiqRXV4lNPaZEAqbVmD2x5VNbU4dPoivvn+AoTlVe06lxBCnlYUmCpgB4EdHTEtflKORnGTfNvW2hL2ttbwZAWm7Mz8hkYxI4Q1LjMAACAASURBVLnExMRIaStS0j0oX2NqmCOmMnOeHQNT09ZlF+WVNfj3p8cYxwQH+jAy9jkjpt2gyH7xk3JGPdamJglu38tUcwZTelYeZ9+l6ylobtZsvalUKsWuL0/ghwvXcSb2BnZ9eYKTbEYIIT0RBaYKlCU/deQ/C/ZoqZe7M3g8HrxYNT3ZNR/ZGfn2ttYGVfqJtI+7qyNMTFr/idnbWcPK0lyPd9Q2Z0d7TB47lLGvWlTH2J7Cet2NNTLcHUZMz12+yanHej/jscbn33+Uy9lXXlnDSBhTe37GY9xLb32/9OwCTsIkIYT0RBSYKrCyNGeMFjU0NnUow5g9RS8rMs8eMc0resIIfNnrSx1pfWm3ZmFuhqnjWlvszmB1WTJUM54ZobLxgcCZj8FhgYx9Hqy1tIUlQoMe/auta8DFa8mc/fcyHmt832mPuCOmAHDxKve6ypyNvcHZd+tOupIjCSGkZ6HAVAGPx9PJOtPcAmYrUlnPeEe+LaO2ZV19IyMY5ZaKMozWlaTj5sWMw871S/HOsucwZdywtk8wANZWFvjrpJFKX5s0ZgiMjJg/NhzsbWBhbirfrq1rRGV1bafeozZiryejrp77C2d5ZQ2Kn7TdUrVRLMajxwVKX7t1N6PNJUBFpWW4mfqQsz8xlQJTQgihwJRFF5n5uawRUy+3lhFTZbUtFUdX2V2f2mpHSroHHw9XuDoZ9tpStujRgyFwZpaBsrI0w7jh4ZxjeTwe3FyY0/kFKhpI6JtEIsEvl7ijlTKaTOdnZBegqYlbgxgAmpsluPwHt8udol+ULCMAgIycAq26zRFCyNOAAlMWbUdMJRIJZ+2obMQUgJKSUQqBaQe6PhHSGUxMjDF3+jjGvgkjB8LSQvkaWQ/WOlPFmqeG5GZqOkqElSpfv5/BXTvKlpbJnMY3N2N2dbt0PVnlkgBRbT1ilSwjAFoy+2/ffdTm+8sIy6vw7Q+x+P7n31FTW9f2CYQQ0g1QYMqibWZ+UWk5xOLWzFx7WytGtyMvTs3H1iCWHQRTqSiiT5ED+2Le9LHwFDghKiIMs6ZGqTzWXcAaMTXQzHx29yovN+a/x3sajJg+YCU+PfeXkYwkt4LiMk7wKhN7PRn1DWKV107UcJ1pc3Mz3vvsKE6dv4ajZ+Kw//tfNTqPEEIMHQWmLOwR0/ZOreWyC+uzMvHZI6aMqfwK5ogpFdcn+sTj8RATPQIf/GMZ/rZgGsxMTVUey24mYIiZ+Vm5RYxMeABY/uI0RlBZKqzEkzLVI6oSiQQPWaWihvTvgyFhfRj7Ll69zTm3ubkZv1y+yTqXmUiWfD8TYoVSc6ok389ETl7rWvZrt+4pbXFMCCHdDQWmLNwR0/YFpspKRSlSN5XPHp2l5CfSXbCn8gsNsPsTu21oWJAfAvw8EODrwdivbtT0cUEpautaE6dsrS3h5ebMWXt7Pek+ausaGPtupqajVGEZgampMf5vzhTGWvL6BrFGo7bs5QCqWhwTQkh3Q4EpCzsYZK/7bAunFSlrxNTVyZ4xQlNZXSuvE0nJT6S7Yic/FQvLNS423xXKK6tx7dY9xr4p41rqsfYL8GHsZ0/VK0rLZL7Wp7cneDwe+vftBSeH1hmOhsYmXE1ktjllt3UdNSQUfDtrDAoNYOxvq2xURZUIiXcyOPuVtTgmhJDuhgJTFntOVr52U/nsovrGxsacuo8FxUJO1ydjY+r6RLoPaysL2CvMNjQ1SVCqZkocaOl+dPzsFfzj/f34/uffNZrC7qhzcYmMTHp3VwcMDGkJCNmBqbrMfHb90qDe3gAAIyMjjI3sz3gt9nrrqOajnAI8YJ0ra2QwOJQ5nZ94J0NtPdW4hFQ0N3On7ZW1OCaEkO6GAlMW9lR+e9aYNjU1o7CUOYXJnsoHAE83bmtS9sgs3466PpHuhTOd38Y60ys37uDY2St4mJWPo2fisHHXN50yHd0oFuPX35MY+yaPHSr/99WnlyeMjFr/rRUUlylNepRKpZyOT/0CvOV/HxcZDsV/shnZhcjJb1kHyh4tDevrB19PVwBAaJAfzExbM/tLhZUqu0BJpVLEXuOuXwW4LY4JIaQ7osCURVnyk6ZJBUVPyhmjMg72NrCx4nbQUVbLlLo+ke7O3ZWdma8+MP39BnOqO/NxEd5+70vcTOEWn9fGlYQ7jLaq1lbmGBPROrppaWGO3j5ujHOUlY16UlaJMoUERTNTE/Tyaj3PxYmP0CA/xjmXriWjrKIa15IeMPYrtnU1NzPlnJd0lztVD7SUqipQsX6XXaaOEEK6IwpMWUxNTWBjZSHflkiknF7hquQWsKfxuaOlQGvBfZn8oidKSkXR+lLSvXAy89UESg2NYtzLyOHsF9U2YOd/j+PAyd/Q1KT9GlWpVMoZrRw/YgAszM0Y+/r6tz2dzx4tDfDzgKkps4bpuEhmEtSVG3fw08U/GFPvHgJH+TICmcGsdaaqukApy/aXyS9+YtCtYAkhRBM6CUzDwsLA5/M5f2bNmqXyHGXHf/XVV7q4Ha11NDM/t5DZitSHtb5UhpOZXyxEWSW7VBQFpqR7ac9U/t2H2Yx6v2w//fYHNu85CGF5lVb3lPIgC3kKSUFGRjxMihrCOS4ksO3AlF2bNKi3F+eYoeF9YGvdOktSLarjBMaKywhkBrHWmaZn53OWEdXWNeB60n3Oe7a+3tihTnWEEGJIdBKYxsbGIi0tTf7n8uXL4PF4mDFjhtrzPvzwQ8Z5c+fO1cXtaI3b/UmzH/bsUlHeHsoDU3dXB8ZatFJhJYpZnXIoMCXdTXum8pNYHY58PFxgbMz8cfQwKx/rdnyJ+4+UF6vXBDsojBgQBGdHbnvYoN7ejH+TjwtKOTMlD1jBal9/b7CZmZpi1NAQxj7FQUxrK3NEDQvjnOfIt4Wfl4BxDrsL1NXEu2hobE0Qc3KwRS9vAeMYSoAihHR3Jm0f0jZnZ+YI4IEDB2Bra4vnnntO7Xn29vYQCARqj9GHjmbmc2qYuikPTM1MTSFwdmC0bbz7MJtxDAWmpLsROLf8wiULxMoqqlHf0MiZNpdKpbh9jxl0zZ0+FrbWVtj11QkIy1tnD6pFdfj86K+4n10KTzcneAqc4O7qBHdXR9jZWDFGHiUSCZ6UV6GgWIiCYiHyi4W4fS+T8T5Tx0UovXdrKwv4egqQnVcs3/cg4zGGhgfJ70Nx5JXHAwL9PJVea9zwAfj50k2lr00YOZDzPGQGhwUw3j/xTjrGKGT6K2b5A8DYyP4oLatCVm7rOXmFTxAW1Evp9QkhpDvQSWCqSCqV4sCBA5g9ezYsLbmJP4refvttrFq1Cr6+vliwYAEWLlwIIyP9L3vlTOVrUMtULG7iZOSzp+wVeQicGIFpHqsGIXV9It2NiYkx5xeuotIy+HkxE4sKS8oY/epNTY0R0scP5mameO/tpfj4mx8YI6pSKXDrTgZusWp3WluZw8PVCfZ21iguLUfRk3K1ywMC/TwQ2Et5MAkAwQE+jMDwvkJg+pA1je/j6QprhbXoinw9XeHv645HOYWM/aqWEcgMDg3E9z/Hy7dlXaBMTU2Qk1+CjGzm9cZGhuNaIrM2KyVAEUK6O50HprGxscjJycFLL72k9rj169dj9OjRsLa2xuXLl7FhwwYIhUKsWbNG7Xnp6Zr1ktbm/DpRNUSi1lHSh4+ykJ6ufPRTpqCkDNXVrVP+fDsr5OeprodowmtivAdbRdkTpKcbVotBbZ99T9cTnp+5iZTxfZ2QmAJxHfMXu8sJdxnHBPXywOOcbPn2XycMhL2VCc5eviUffVX2b0UkEqGkVPMOU/0DPdR+DazNme9z9WYKhvf3AwBcunqD8ZqTrbnaawX5uiDlHjOQHhjsh7InxSh7Uqz0HKlUChMjCSqrW5YQiETAuYu/I6i3J05e+IPx/n383FFZVgqJuI6xP/V+OtLT/TnXNuTvvcDAwLYPIoT0GDoPTPfv349BgwYhLIy7jkrR2rVr5X/v378/JBIJ3n///TYDU21+iKWnp2t0fkFZPS4mtJZ3MbewbvO8ksq7sLZuHWkNCeqt9pwhwjokpGarfH1geChnSYE+afrsiHI95fmF9s1GTmFrLVLj/9/encdHVd39A//cZbask2SSyUYSCFnZV0FWAXkEqwJqRahFELUqtlV5RKyl+jx9VEQp1SqKKPwopGrZCiIurcgajCAKEoWwypZMMtknme3e+/tjkiF3tmyTZCb5vl8vXu3ce+bOmesk+c455/s9yhC39/3hZ9/IflYmjxvh1iYzMxMTbhyOtzZ+jAs/X5W1b4tRQ7Jx9+03+6wNHBefiH9+9o3zcWWtBYlJvRAaokbljkOyPowbNcznf8/EpF74d34hrE02Dfj1XdN9jtgCwPhRQ/DloetT9qXVFvxXWm+cvvix7PXvvPUmZGRkIDQiGh/s/tp5vM4quvWrp3z2CCHdg18D09LSUnzyySd49dVXW/3cYcOGobq6GgaDAXFxcf7sVqu5Z+U3n/zU3I5PrlxrmTZFuz6RYOVaMuqqQT61bLZY3TLeB+e6j/ABQG5GKl7/0yPYs/9rKEPCcc3gWDt61VCOayXlsqCvUUSYxrkGNTEuGon6GPRKjHXbMtWTyPBQJMfHOJfVSBJw+vxl9MtMxblL8ml0T4lPTYWGqHHXtLHI2/EVAGDcyP7NBqWAYzq/aWB69IczyOrTS5aIFRaixohBmQAAvU4Lnmed9ZOrqk2orav3WD+ZEEKCgV8D07y8PKhUKtx5552tfu6JEyegVqsRGemeMdvZ3LPym09+cg1MvZWKauQrMKVdn0iwStD7Lhl18vRF2SYUep3WLZu/KZZlkZKocxvxkyQJxopqxxKa2jrExkQiIS5GVqqpLXIzUmXrvX88cwkqpULW57iYSERrm18DfvvNozEwpw/qzRaPpaU8GZDdG0oF7wy6S41V+PDjvbI240b2h1KhAHB9i+OmO0VdKTa2+PUIISTQ+C0wlSQJGzZswKxZsxAWJg/s1qxZg3fffRfffOOYJtu9ezcMBgNGjBgBjUaD/fv346WXXsK8efOgUqn81aU2cxsxrWl+xNQtI99Lcf1GoSFqREWGue34BFBGPgleCS4jk8Wl5ZAkyflFyzUbf0i/9DZ9CWMYBrroSI+ln9ojO70XPt//rfNxYdFFaNTyLPqsZkZLm/axd6/45hs20bgLVNNEr2KXUnI3jR4se5wU7xqYllFgSggJWn4LTPfv34+zZ89izZo1bueMRqNs8b1CocDatWvxhz/8AaIoIi0tDUuXLsWDDz7or+60S0RYCFiWgSg6Mi9MdRZYbTbnKIUrq82GkjL5Hw9fGfmNkuN1ngPTCMrIJ8EpWhsOlZJ31ts01VlQVVMHbUQoJEnCMZfA1Ns0flfJdSm0f+5SMXiekx3L7uCgb2i/vm4VCBqlpyYgNUm+1MnTFseEEBKs/Fabafz48aisrMSwYcPczi1duhSVldcTIqZMmYL9+/fj8uXLuHr1Kg4dOoRHHnkEPO/3XKw2YRjGLfGoysd0/tUSo6yIdlxMpNdahU0lxXuezqcRUxKsGIZx35q0YZ3plRIjSpuUiVIqeORmpHZq/5oTFRmO+Ngo52NBEPHjGflWpNl9U1yf5ldDXbYnbWrS6EFux5L0LjvJUWBKCAliXV80NEC1ZlvSptNoQPOJT428japSYEqCWUJclOxxcUNJp2Mn5aOAuRkpUCk9z0J0pRwfgWd4qAZJPtaH+0NMVIRsF6hGSgWP0UNz3Y572uKYEEKCFQWmXrRmW1K3rUhbGpjqKTAl3Y/r1HJjoPS9yy5MgTaN38hXYJqVntwpiYnDBriPmo4amuOxqH+iPlq2nWpZeRUsVltHdo8QQjoMBaZetCYz371UVPPrSwFfI6a0xpQEL9fSTNcM5Y4yUWflZaKG9AvMwDS3r/fkpqw+LUt8aq9h/d3rjnqaxgeub3HcSJKuL58ghJBgQ4GpF62pZXrZtVRUYstGTCPDQxDmYQTENSgmJJgkuqwxLTaU44dTF2Qll+Jjo1pUW7QrxMZooYuO8Hgup4UZ+e3VJyUBvZp8wU1L1vusneo6Sn3pGq0zJYQEJwpMvXBNfqqs8TxiWm+2yPb9ZhjfNUqbYhjG46gpTeWTYOZal7S4rAJHTsi3xAzU0dJGnqbzlQq+1eWf2ophGPz+gVkYPTQbY0f0w5MLZ/lcQuCaSEkJUISQYBUYafAByDU49DZi+v2P8nVzCXHRXstKeZKoj8Gpc5edj2nXJxLsQkPUiAwPQVVNHQDAbhdx6OhJWZtAXV/aqF9GCvYX/CA71jct0a10VEdKjtfh9wtmtbhtU1cpAYoQEqRoxNQL1+l0b+WiDh/7UfZ4SD/vpV48cV2PSrs+ke7AddS0sa4p0FgmqmNLLrVXdrp7/5rbhrQrUckoQkh3QYGpF+67P7kHpharDcdOyguGjxqS3arXcR3poGl80h241jJtql9maqtmFbpCfGyU289iIO+m5Lp86FppBex2oYt6QwghbUeBqReuI6YVVTWQmlbRh2N7RbPlelmWaG04MtKSWvU6OX1TEBl+fep+xMCsNvSWkMDiOmLaVKBP4wOONZ43jx3qfJyojw7oUd7GLY4bCYLothudq71fn8Djf3oTf/rLhmbbEkJIZ6E1pl6oVUqoVQpn4Gm3izDVmxEWonG2+frYT7LnjBqc3eppeJVSgf958tf4z6HvoI/RYvKYIe3vPCFdLNFnYNqnE3vSdnfcPBrR2jAYjFWYOGpgwI/yum5xfLm4zGtJulJjJdb8YxfsdhEGYxVeefsjvLzkASgU9CeBENK1aMTUB1+1TK02G47+IM80vqGV0/iN4mOjMfeOSZgydiitLyXdQoKXyhSJ+uiALRPliuc53DR6MO75xQRZndBA5ZqZ7ysBav83P8jKd10uNmLLpwda/FqFRRfxtw07UG+2tL6jhBDiAwWmPriVjGqSmX/8x/OyafyoyLCAXoNGSGfSx2jh6TvW4JzAn8YPVq6jo5e9JEBJkoR9BSfcjv/ri3ycv1Tc7OvUmOrxxv/bgf0FP+CZ5e/j7MWrbeswIYR4QIGpD64JUE0z8w9/J5/Gv2FwFo12EtJAoeARF6N1Oz44wOuXBjO3rWC9BKZnLlzFNYP7mlJRlPDWxp0+k6YkScLqjTtRXlkDACgurcCfVv0dxorqdvScEEKuo8DUB7dapg2Z+TabHUdPnJadu2Fw26bxCemuXBOgVEoeOT62+yTtkxwv33HuaonRLWETgMfR0kY/XynFv77I93r+071HcPTEGdmx6RNHIibK805ZhBDSWhSY+uC+xtQxlX/i1HnU1VudxyMjQgO6xiEhXcF1a9J+mWkBn0AUzFy3OLZY7Sgrr5K1sdnsOHS0UHYsUS//ArH1swO4eMXgdv0Ll4uxcft/ZMcy0hLxy1vHt7frhBDiRIGpD261TBum8g+7ZOPfMCgLLEu3kpCmBrlk348d3q+LetIzeNri+IpLAtS3J8+gts7sfBweqsGy386Vlayz20W8veljCML1KX2zxYpV72+TJUyFaJT47fwZnbobFiGk+6NoygdPyU92u4AjrtP4bczGJ6Q7G5TTB3NnTEJWn2TMvm0CbhyW29Vd6vZc15m6JkC5brM6ZnguoiLD8cA9t8iOn/u5GDv/87Xz8fsffea2LvWhe2/1uI6YEELag4rW+eBpKv+H0xdgqrteIiUyPAQ5NI1PiBuGYXD7lFG4fcqoru5Kj+FaMupKcRmyejmm6qtr63CsUL4+dPzIAQAca+RHD81G/rfXZ4P++ck+DB+YiXM/X8Per+XrUiePGYzRQ3M64i0QQno4Ckx9cA1Mq6pNyP/2R9mxEYOywHE0lUUI6XquWxxfKb4+lZ//baFsKj5RH40+KQnOx/Pv/i+cPH0R1bX1ABxT+n9dtw2GskqX14jBvDtv7ojuE0KIf6byX3rpJWi1Wtm/zMxMn885efIkpk+fjvj4eOTk5GD58uUeM0i7UkRYiOxxdW0djhynbHxCSGByDUybZua7ZuOPHzlAVuIuMjwU9981Vdbm5yulsnrNSgWP3y2YCZWSktgIIR3DbyOmGRkZ+Pjjj52PfY0iVldXY+bMmbjxxhvx5ZdfoqioCI899hhCQkLw+OOP+6tL7cbzHCLCNM4RBEmCW+JAvwDeP5sQ0rPooiOhUvKwWO0AHMXwTXVmXCkuw5kL12Rtx43o7/b8G4fl4tC3hThyvMjtHADMu3MKUhLj/N9xQghp4LfAlOd56PX6FrX95z//ifr6eqxevRoajQa5ubk4ffo03nrrLSxatCigCtVrI8KcgamrEYMyaRqfEBIwGIZBQlwMLlwucR4rNlbh1KVyWbv+manQRUd6fP7Ce27BT2cuyb6EA45NRCaPGdIxHSeEkAZ+y8q/cOECsrOzMXDgQCxYsAAXLlzw2ragoACjR4+GRqNxHps8eTKuXbuGixcv+qtLfhHpUjKqKZrGJ4QEmmSXBKji0grs/0aejd+Y9ORJVGQ4fu2yhjQ2JhIPzZkeUIMGhJDuyS8jpsOHD8dbb72FjIwMlJWVYcWKFZg6dSoOHz6M6Ohot/YGgwGJiYmyY7Gxsc5zaWlpXl+rqMjzFFNLtfb5dks9TCaT2/EQtRIq1tbu/gSTnvReOwLdv7aje9dyLOyy31mHvzuNKyXXSz0peA7RYbzPe5oQpcKYIX3xZf4JRGvDcPfU4bh25XKH9DcjI6NDrksICU5+CUxvvln+7Xr48OEYPHgw8vLysGjRIn+8hFN7fokVFRW1+vl903/Gj+dL3I5PHDUQOdk9Z8S0LfeOXEf3r+3o3rVOuUnAviPXkzSvlFQgNPT6zM/Y4bkY0L/5mrKZmZn47QO/7JA+EkKINx1SYD8sLAzZ2dk4d+6cx/NxcXEoLS2VHWt8HBcXWAvrXUtGNRpFRfUJIQHIdfcnV+N8TOMTQkhX65DA1Gw2o6ioyGsy1MiRI5Gfnw+z+fri+j179iAhIQGpqakd0aU28xSYhoaoMCCrdxf0hhBCfIvXRYHjPP9qj4oMw4CstM7tECGEtIJfAtPnnnsOBw4cwIULF3DkyBHMmzcPdXV1uPfeewEAL7zwAm6//XZn+7vuugsajQaPPvooCgsLsWPHDqxatQqPPvpowC2u13pIfho2IJP2hyaEBCSe5xCvi/J4bszwflRJhBAS0PyyxvTq1atYuHAhjEYjdDodhg8fji+++AIpKY4an8XFxTh//ryzfWRkJLZt24bFixfjpptuglarxWOPPeb39aj+4GnEdBRl4xNCAlhSfAyulBjdjvvKxieEkEDgl8D0/fff93l+9erVbsf69euH3bt3++PlO1R0ZDh4nnVu5adRKzEwm6bxCSGBKyleB3wv36UuNTkOqUmBtYafEEJcdcga0+4kRKPCLRNGOB/fN3MyFAq/7UtACCF+l6iPcTtGo6WEkGBAEVYL3DdzMsaPHIBQjcrjbimEEBJIeiXEyh4zDDB2eL8u6g0hhLQcjZi2UGpSHAWlhJCg0CshFglx1xOgxgzv57X0HSGEBBIaMSWEkG6G5zksfuhu7Prya9SbavDg7Gld3SVCCGkRCkwJIaSNWPMZMGIdBFVfgAvp6u7IJMfr8PCcW1FUVAS1StnV3SGEkBahwJQQQlpLkqCs2ALeVOB4zChgDx0BW9hYSArfOy8RQgjxjgJTQghpDUmConLH9aAUACQb+NpD4GvzIWj6wxY+AaIqpev6SAghQYoCU0IIaQVF9edQ1B70clYCV38CXP0JCKresIdPgKDOcaTFE0IIaRYFpoQQ0kJ89V4oqv/Torac5Tw4y3mIfBys0XdBVKV1bOcIIaQboHJRhBDSAnztYSirdsmOSawGZv1vYYmeDVGR4PF5rN0AdelaMNZrndFNQggJajRiSgjpeUQLGHsZWHsZIAkQlSk+k5a4uu+grNgmP8goYdE9AFGZDCiTIYQMAWspgqJ6LzhLkbytZIW6bD3q9Y8DHNUTJYQQbygwJcFPrAMj1EHiY2gtH5ET68BaLoG1l4K1l4KxlYK1l4ERKt2b8nEQNP0gaHIgKlMAxjGhxNUXQmX8AIB0vTHDw6y7X57gxDAQ1ZmwqDPBWK9AUbMXfN13108LFVAZ/w5L7IMAQ796CSHEE/rtSIKW44//PvB1xwEIEDQDYImZ6wwoSM/F2AwNn41vAcneouewdgPYGgMUNXsgsaGOAFWRAGXVbgBik5YcLDG/gqju6/VakjIJ1pg5AKsBX5t//ZmW81BWbIc16k76EkUIIR5QYEqCiyR5nS7l6k+Arz0Ee/jYLuoc6VKSBNZyHoqafeDMhe26FCOawJuOeDoDS8w9EDS5LbqOVXsbGJsBnOWs8xhvKoCoiKfPKSGEeECBKQkOkh1c3XEoavaCtXlPIlFW7YagzqYi5z2JJIKrPwFFzT6w1kstfBIDiY+GyMcCkg2c5QIAodlnWaNmQggZ3PK+MTwsMfdBY3gDjN3oPKys3AlRoYeozmj5tVpDsoE3HQNnLkScaAJjj4PER3bMaxFCiB9RYEoCm1gPvrYAitoDYISq5ttLNigrNsMS+zBNlXZ3ogW86Yjjs9Ek6HMlcVEQ1OkQ+VhIfCxERWzDeuQmv/7EenDmU+DqC8GZT4ER692uY428FfawUa3vJxcCs24e1CVvgpEsjb2CyrgR5rhFkBSxrb+mN0IdFKZ88DWHwIg1AIAIyQS1YTXMcY9QcEoICXgUmJKAxNgrwdceAF/7dZM/5u4EVR+IiiQoavc7j3GWc+BNh2EPG90ZXSU+MHYjWOsliMpUSHyUfy4q1EDRsMsSI9Z5bSYqk2ELnwBB0x9gON/XZDUQQgY7RkMlO1jLBXD1PzqCVMkKW8RN7fo8SYp4WGPmQFW2Ho1JVIxYD1XZepj1iwBW0+ZrAwBjL3esqTUdASSr+3mhHKrSd2GOewTguoa5twAAIABJREFUQtv1WoQQ0pEoMCUBhbFebUha+R7ep1YZ+baPkgTWbgBnPuVsoazc5ZjS91cwRFqFtVyAomYvuPpCOAIxDrbwMbBF3AywqjZdk7GVNnw2jvpMaBLUOQ2fjd5tGzVneIjqvhDVfWHDbW3qq8d+aXJgjZwGZdUnzmOsvRQqYx4suvmtT9qTJLDWS+Br9oGvPwFZ1QAPWLsB6rL3YI59sN2BMCGEdBQKTEnXkySwljNQVH/lXv+xKUYJe+hw2MLGyteQMgysUbOgKX7t+miRZIWyfDMssQtpSr+zSCK4+pMNaz0vupwUnF84rNpfQNAMbNl/F0mCWroCVdnBhiDXGw720KGwhY+HpNC35110KHv4BLC2Yke1gAac+RTUxa9C0ORC0ORCVKZ6H+FtMprLmwt9LmEAo4TI6wBc/5lirZehKlsPi+4BgFX66V0RQoj/+CUwXblyJXbu3IkzZ85AqVRi+PDh+NOf/oTcXO+ZqxcvXsSgQYPcjm/evBlTpkzxR7dIoHMmNO0Da7vqvRkbBnv4GNhCRwNciOc2fBSs2ltlRdA5SxF40zewh430cmEJXN0xKGoPghFMsEZOhRA6tF1vqUcSreDrjkJRs893oASAEaqgMm6CoP4GVu0Mz0lqkgTWdtkxlV5/EsniGXD1nqefJVYDe9ho2MNuhMRF+OPddCyGgTX6TrD2MrDWn52HWXsZ2Jp9UNTsg8SGQFBnOwJVdRYAsdn1r01JXARsYWNhD7sBYHjUVb6GUJQ7z3OW8456qrp5VE+VEBJw/PJb6cCBA3jggQcwdOhQSJKEF198ETNmzMDXX3+NqCjfU6lbtmxB//79nY+ba0+6AdEMvvbrhoDQvdC5sxkfC3v4eNhDhwKMotnL2kNHgas7LivNo6j8GII6ExKvlbVlbCVQVmyXtVWVfwCrJMAeNqINb8oDSQRX/wP4uu+hFyvB2CICejSvRUSLo0C9vRSszQDGXgbOXARGNHl/DqMAJJvsEGc+DU3xa7BFTIQtYlLDsTMNwdePYIRqn92QuCjYwsfBHjqizUsDugyjgEX3a6hL3vCY0MeIdeDrvm0YVW0cOW2+YoCoSIAtfDyEkEGygPMaeztilHtko9ic+RRUxn/AEjOn+fW3hBDSifwSmG7dulX2+J133kFKSgoOHz6MadOm+XxudHQ09Pog/2NNWuR6QlMBGMnstZ2oSnMkrahzWzcNzzCwRt8FTfFfnFP6jGSGsmILLLoFjmuJFiiq/w1FzQF4+mOvrNgMiVU6/ri3lWgBbzoKRe1+5whiuGSCpuR1WKJmQQgd1vZrdzLWchF83TEwtpKGHZNaUBmhgaDOcqz1VPaCovo/UNTsh/yeC1BU/we8qQCMaHYLXj1pVUJTAJO4CJjjHoGyYkfD2mhvgWdzASkHQZ0BW9gYiOpMjz8vEqOAOXY+1IY1spkJrv4ElBVbYI26u/XLXRrWt4rKXrRUhhDiVx0yj1NbWwtRFKHVaptte99998FsNiM9PR2PPvoo7rjjjo7oEulAjL0SrOUcvP0RZSSAtZxtXUJTG0l8DKyRt0BZucN5jDOfAld3FGBUUFbu9DlK6yjj8wEsjBKCJqd1Ly7UQFGbD772kOdscckGVfmHsFvOwRo1o0WjwF1GqIay8hPZWsiW4WAPHdKw1jPeedSmnQ576DAoK7aBs5yTPYMRapq5JgNBnQ1b+HiIqvRuEwhJfDQssfc3lKo63TBa/FPzU/UNu1IJ6hwI6syWjRizITDHPgC14W2w9lLnYd50BBKjgk17a8um9Z3riPeCtf4Mi24BBE12888jhJAWYiorK32ncrbB/fffj7Nnz+Krr74Cx3ke1TAajcjLy8OoUaPA8zw++eQTvPbaa1i9ejXuuecer9cuKvKRHEM6FSNZES0dhlY6BqaZjGBvJPCoZnJRyQyFjWn+i0zLLiohSfwnNLg+OiSB8drHeiRCjWuy8xJ4XGFnwMwkN/tyCqkCWulbREiFYFow5QoAFuhQzN4KG9PM0hVJAod6iOAhMZ2QrCKJiJSOI0Y6BBbuZYe8EaFCFTMAlcxgCEyYj+tLCJdOQSftAwcfpZ6gQB2TChP6oI5Jg8B4Xlvc7Ugi1LiKUOkcwqRzUMDxJcqKaJiYPjAxfWBGfJu33eWlGiSLH4GH/MuACBXqmFTUMn1QhzSIjFp2npFsiJAKoZWOOfsEAPXohSvcnW3qS6OMjA7aZIAQEpT8Hpg+++yz2Lp1Kz799FOkpaW16rlPPfUU8vPzcejQIX92yamoqIh+CbaR7N5JErj641BWftyqqd2mJDYM9rAbYQsb3SF1FRlbGTQlf/E5PSxxWli1t0HQ9AdXdwyq8g8hK7nDKGGOe9gxXen2ZAmstbEk0o/y58lwEDS5qC87grBQ+ciWxKhgjb7LfdmAZAdruQiuvtCZeS0xalij7ujQZQCs5WcoK7b6TEQDWIh8dJNC9TqIfBxEZXLrsrzFeiiqPoei9hAa753EaSFocmBX50BUpztHlHvsz60kOYvktyexy/X+MbYyqA2rndd2x0JQ9XZUCFD1AVd/0vssAACz/reO//6EEOIHfp3KX7p0KbZu3YqdO3e2OigFgGHDhmHTpk3+7FLwkyTwNV9BUZsPkY+FNWpm67bblOxQVO4GX/89RF4Pe/iNrV+72QRjK21IGmrbyLUjoWkc7KHDOnQqW1LoYI2YCmXVLg9nOdjCxzuSbhqmQYXQobBKFllWPyQrVKXvwRz3m+vT0i5TmV5fn9XAHjoK9vAxkLgIXCrviyz+gGwalZEsUBk3wW45D2vEFHCWs+DqT3rMvGYkM1TlH8Iq1MIeMaHN98UjoQ7Kqt3gTV97PC0qEmCLmARRkQCJj/ZPJjergS3qDtjDxoC1XoSkiIeoSOw20/R+wTAdUmlAUuhgjl0IdenbXpYNiI7PYpPEQG8EdTYkBO9aX0JI4PFbYLpkyRJs27YNO3fuRGZmZpuuceLECUqEcsHX7IOyajcAgBMqoTa8BXPsA5CUSc0/WTRDVfb/nH9gOKEanKWo1dnugGMqT1H1KRTVe+FpnajEaSGo+/q4gAKCOrNdQXFr2cPHga8/IQsgBVU6rFEzPGbH28NGA6JFVgCdEeugLn0XZt1CcNYLzZZE8pYtbmV0MOsfh7JiS8Na2+v42kPga1s2S6Cs2gVGrIEt8tb230fRAt70DRTV//Y4GiYxKtgip8IedmOHJRpJCh2E1nzRIn4hKRNg1v8eiup/g6v/EYxY24pne15HTAgh/uCXwHTx4sX48MMPsXHjRmi1WpSUlAAAQkNDERbmWG/2wgsv4OjRo9ixw5GUkpeXB4VCgYEDB4JlWXz66adYu3Ytnn/+eX90qVtg63+SBUkAwIi1UBvehkU3D6KvQFCogbr0PY/Tsqy9FMqKLVBUfea9PqgkghEqwNhKwdoNSBF3Q1Htaf2k++hjwGBYmHX3NyQ81cIeNgKCZpDPgM4eMRGMZIai+svrlxFqHMsCfBAVSbBFTICgGeA9iGPVsEbPgajqA2XFDrSkBJAnipp9YIRaWKPvatvopVDdZEtPz4k29pDBsGl/ERy1QUmbSHwUrNF3A5II1nqpIfmqEKytxHN7l1kAQgjpCH4JTNeuXQsAbhn1S5YswdKlSwEAxcXFOH/+vOz8q6++ikuXLoHjOKSnp+Nvf/ubz8SnnoSxlUBlzIOntYuMZIG69D1YYu6FEDLQ/bzdCHXp2uaLnYu1UFR9BkX1HthDh0Fi1A01Kg1g7UbZto82mADI14L6Gn0MGFwYrDH3tuoptoj/AiOaWzSK2epscYaBPWw0RGUyVGUbwQgVHptJbEhDgfUcgFFAadwERrI4z/N134IRTbDE/KrFXwgYm6FhP/Wj8BYUO5aLzICo7oFrOnsqhoWoSoWoSoUN08DYjc7NDTjLBUi8FrawG2EPHRl4Xz4JId1Oh2TlB6qgSaIQ6qAxvNFsYAkwsEbNcExBNx6xXoW67D23EjyishdEPraZkk3emUwmhIY6AlOJC29IGvI9+hjUJAnK8o8c+7K7af1UpsfPnlgHVfkWcPUnHA/5uCbbUqbIMq8Z6xWoS993S1gRlSkw6xZ43RELktiwb/0+cGYfW3oyClgjJsMePj7gdgMKmp/bANWu+ydJ3fdnnBASkALrLxABJAEq4ya3oNSq/YUjWJIl80hQVmwDI9TCFjEFrOU8VGXr3YrXC+os58iaLfIW8LUHoKg9fH1f+RZjYAsbA1vkzQCradv7CxYNxfoh2cDXHwfQAVOZbAgsuvvA2MoAhoPEey8dJSmTYI57xPGlo8lng7X+DLXhLVhi5oKRLA07MpWBtRsalmEY4euLSOOWnrawGwGaniWuKCglhHQyCkwDjKLyY7eMd3vocNjDxjVk6YZCVb4ZgHj9OdVfgLVdAWc+LZt+BwB7yBDHOrKGUTCJ18Km/QVsEZPB1xY4difysv2jxIY6ywKV1dnBxk+BpIj17xsOZAwHa8xc2C0jwUg2CKq+HTKV2dIqC5JCh/q4R93WDrN2Q7NrYN2uFcxbehJCCOm2KDANII37xzclKtNgjZrlHLkQQofDwoZCZdwoq9HJ1btP09rCxsGm/YXnUQ9WA3vEBNjDxzj2l7f+DIlRNdSmjIXIx8qmhyuNRYjtSUFpI4ZxbPUYKLhwmON+I6u20BqispdjP/Ug39KTEEJI90SBaYBgzeegrNguOyZxWph197mt+RM0OTDHPghV2TqvWdXWyGmwh09sfiqO4SGEDoUQOrQ93SediVXDErsAKuMHzvWpvkhsGERVb8d+6qreND1LCCEkYFFg2hlEC1jLBVlWtYwkNOzt3mQtIKOEWTcP4MI9X1KV5lhzWPqey+5LLCzRd0IIHeG37pMAxChgiZnrqC1bs9+RWc3rHDsxKWIh8nHO/9/t1wMTQgjpNigw7SCMvRKc+ceG2oBn0NpMeEv0Pc0W0ZcU8TDHPQpV2XqwtmuObStj7oGg6deOnpOgwbCwaafDFjmt4TGNhBJCCAluFJj6iySBtV1uqP9X2Mx+477ZIm6GEDKgZS/LR8Gs/x1Y2zWIXJT3skGk+6KAlBBCSDfRowLTCPE4FBWFjjqRqt7+qdcomBy76JgKXKbU23g5zQDYIqa07kkMC7ElW5QSQgghhASwnhWYSiehqK2FovYgJEYNQZMFQZ0LQZMFsK0baWRsZVDU7gdvOiLLjvdG4mMgKhJ9thGUybCHj6MRMEIIIYT0SD0mMGWEaqhRgsZtNRnJDL7u+4adkFgIqt4Q1JmQFHENiSNRHkdUWctFxy469T/A03ahTV4RojIVdk0uBE0OJD6OAk5CCCGEEB96TGDK1Rc2KUnvSgRnOetSF5KFyEc7C8xLnBZ83XGw1gveX4RRQlBnwa7JgaDOBrgwv/WfEEIIIaS76zGBqT1kCK6xlUgLrQZX/xMYsbaZZ4hg7WWAvQyc+UefLSVOC1v4WNhDRwKs2n+dJoQQQgjpQXpMYApWBROTDmt0BiCJYK2XGko5FYK1lbTpkqIi0bGLTshA/yRSEUIIIYT0YD0zmmJYiKpUiKpU2DANjK0MnPkUWFsxGHsZWHup1/3jAUBQZ8EWPh6iqi+tGyWEEEII8ZOeGZi6kBQ62BU6+UHR4ghQbaWO/7WXQ+LCYA8ZBkmZ0DUdJYQQQgjpxigw9YZVQVQmA8rkVu7ZRAghhBBC2oLt6g4QQgghhBACUGBKCCGEEEICBAWmhBBCCCEkIPg1MF27di0GDhwIvV6PCRMm4NChQz7bHzhwABMmTIBer8egQYPw/vvv+7M7hBBCCCEkiPgtMN26dSueeeYZPPXUU9i3bx9GjhyJu+++G5cuXfLY/sKFC/jlL3+JkSNHYt++fXjyySfx9NNP41//+pe/ukQIIYQQQoKI3wLTN998E3PmzMG8efOQlZWFFStWQK/Xex0FXbduHeLj47FixQpkZWVh3rx5uPfee/G3v/3NX10ihBBCCCFBxC+BqdVqxXfffYdJkybJjk+aNAlff/21x+cUFBS4tZ88eTKOHTsGm83mj265ycjI6JDr9gR079qH7l/b0b1rH7p/hJBg4pfA1Gg0QhAExMbGyo7HxsbCYDB4fI7BYPDY3m63w2g0+qNbhBBCCCEkiFBWPiGEEEIICQh+CUxjYmLAcRxKS0tlx0tLSxEXF+fxOXFxcR7b8zyPmJgYf3SLEEIIIYQEEb8EpkqlEoMHD8aePXtkx/fs2YMbbrjB43NGjhzpsf2QIUOgUCj80S1CCCGEEBJE/DaV/9hjjyEvLw8bNmzAqVOnsGTJEhQXF2P+/PkAgIcffhgPP/yws/38+fNx7do1PPPMMzh16hQ2bNiAvLw8LFq0yF9dIoQQQgghQcRvgemsWbPw0ksvYcWKFRg3bhwOHz6Mjz76CCkpKQCAy5cv4/Lly872aWlp+Oijj3Do0CGMGzcOr776KpYvX4477rjD62scPHgQs2fPRk5ODrRaLTZt2iQ7bzAY8MgjjyA7OxsJCQm48847cfbsWbfrHD16FDNmzEBSUhKSk5MxdepUWcJVZWUlHnroIaSkpCAlJQUPPfQQKisr23uLulR7793Fixeh1Wo9/nv99ded7SwWC/77v/8bffr0QWJiImbPno0rV6502vvsKP747JWUlOChhx5CZmYmEhISMGbMGHz00UeyNvTZ83zvzp8/j7lz5yI9PR29evXC/fff75ZY2R3v3cqVK3HTTTehV69eSE9Pxz333IPCwkJZG0mS8NJLLyE7Oxvx8fG49dZb8eOPP8ratOTenDx5EtOnT0d8fDxycnKwfPlySJLU4e+REEKa8mvy08KFC3HixAkYDAbs3bsXY8aMcZ7btWsXdu3aJWs/duxY7Nu3DwaDAcePH8eCBQt8Xt9kMiE3Nxcvv/wyNBqN7JwkSZg7dy7OnTuHTZs2Yd++fejVqxfuuOMOmEwmZ7sjR45g5syZGDt2LL744gt89dVXWLRoEXiel72P48ePY/Pmzdi8eTOOHz8uG+0NRu29d8nJyTh16pTs32uvvQaGYXD77bc7r7V06VLs3LkT7733Hj755BPU1NTgnnvugSAInfp+/c0fn73f/OY3OH36NPLy8pCfn4/Zs2fj4YcfxsGDB51t6LPnfu9MJhNmzpwJSZKwY8cOfPrpp7BarZg9ezZEUXReqzveuwMHDuCBBx7AZ599hh07doDnecyYMQMVFRXONn/961/x5ptvYvny5fjyyy8RGxuLmTNnoqamxtmmuXtTXV2NmTNnIi4uDl9++SVefvllvPHGG1RXmhDS6ZjKysqg/EqclJSEV155BXPnzgUAnDlzBsOHD8f+/fsxYMAAAIAoisjMzMSyZcvw61//GgAwdepUjBs3Dn/84x89XvfUqVO44YYb8Omnn2LUqFEAgPz8fEybNg3ffPNNt6gJ2NZ752rGjBlgGAbbtm0DAFRVVaFv375488038ctf/hKAY6R8wIAB2Lx5MyZPntwJ767jtfX+JSUlYfny5fjVr37lvFb//v3x8MMP4/HHH6fPnpd79+WXX+LOO+/E+fPnodVqATg+a2lpadi2bRsmTpzYI+4dANTW1iIlJQWbNm3CtGnTIEkSsrOz8eCDD2Lx4sUAgPr6emRkZOB///d/MX/+/Bbdm/feew/PP/88Tp8+7fzysGLFCrz//vsoLCwEwzBd9p4JIT1LtykXZbFYAABqtdp5jGVZqFQq5OfnA3Bk/RcUFECv1+OWW25B3759MW3aNOzdu9f5nIKCAoSFhcmStkaNGoXQ0FCvmwUEu5bcO1cXLlzA3r17cf/99zuPfffdd7DZbLKNE5KTk5GVldVt7x3Q8vs3atQobN++HeXl5RBFEbt27YLRaMSECRMA0Gevkeu9s1gsYBgGKpXK2UatVoNlWWebnnLvamtrIYqiM0C/ePEiSkpKZD9zGo0GN954o/N9t+TeFBQUYPTo0bIR7cmTJ+PatWu4ePFiZ7w1QggB0I0C08zMTCQnJ+N//ud/UFFRAavVilWrVuHKlSsoKSkB4AimAOCll17C3LlzsWXLFowePRqzZs3CiRMnADjWu8XExMhGCBiGgU6n87pZQLBryb1ztWHDBuh0OkyfPt15zGAwgOM4t3JfvjZa6A5aev/WrVsHhmHQp08fxMXF4aGHHsLatWsxcOBAAPTZ83bvRowYgbCwMCxbtgwmkwkmkwnPPfccBEFwtukp9+6ZZ57BgAEDMHLkSABwvn9fm5u05N542/Ck8RwhhHSWbhOYKhQKbNy4EefPn0fv3r2RkJCA/fv34+abbwbLOt5m43q0+fPn47777sOgQYOwbNkyDB06FOvWrevK7neplty7pux2OzZt2oR7772XSnuh5ffvz3/+M4xGI/71r39hz549ePzxx/HII484vxT1RC25dzqdDuvXr8cXX3yB5ORkpKSkoKqqCoMGDfL4+eyunn32WRw+fBh///vfwXFcV3eHEEI6BN98k+AxePBgHDhwAFVVVbDZbNDpdJg8eTKGDBkCANDr9QCArKws2fOysrKcFQPi4uJgNBohSZJzhEGSJJSVlXndLKA7aO7eNbV7926UlJS4rT2Ni4uDIAgwGo3Q6XTO46WlpRg9enSHv4eu1Nz9O3/+PNasWSNbSzlgwADk5+djzZo1eOONN+iz5+OzN2nSJHz33XcwGo3gOA5arRaZmZlIS0sD0P1/bpcuXYqtW7di586dzvcMXP+dVlpail69ejmPN93cpCX3xtuGJ43nCCGks3TL4YbIyEjodDqcPXsWx44dc043p6amIiEhAUVFRbL2Z8+edf5SHzlyJGpra1FQUOA8X1BQAJPJ5HWzgO7E271rasOGDRgzZgz69u0rOz548GAoFArZxglXrlxxJl/0BN7uX11dHQC4jXRxHOccyafPXvOfvZiYGGi1WuzduxelpaWYNm0agO5975YsWYItW7Zgx44dyMzMlJ1LTU2FXq+X/cyZzWbk5+c733dL7s3IkSORn58Ps9nsbLNnzx4kJCQgNTW1I98eIYTIcM8888zzXd2JlqqtrcVPP/2EkpIS/P3vf0dubi4iIiJgtVoRGRmJ7du3w2AwQJIkHDx4EAsXLsT48ePx1FNPAXCsq2JZFn/961/Ru3dvKJVKvP/++/jwww+xatUq6PV66HQ6HDlyBJs3b8aAAQNw5coVPPHEExg6dGhQl55p771rdOnSJTz99NN47rnn0K9fP9k5tVqN4uJirF27Fv369UNVVRWeeOIJRERE4IUXXgjqadf23r+oqChs2bIFBw8eRE5ODiwWCzZu3Ih169Zh6dKlyMjIoM+ej8/exo0bYTabYbVa8fnnn2PRokVYsGAB7r77bgDotvdu8eLF+OCDD7B+/XokJyc719gCjh33GIaBIAhYtWoV0tPTIQgC/vCHP6CkpASrVq2CSqVq0b1JT0/HunXrcOLECWRkZCA/Px/Lli3D73//+6AP7AkhwSWoykXt378ft912m9vxe++9F6tXr8bbb7+NN954AwaDAXq9HrNnz8bTTz8NpVIpa79q1SqsXbsW5eXlyM7OxrJlyzBx4kTn+crKSjz99NPYvXs3AGDatGl45ZVXnJmwwchf9+7FF1/EmjVr8NNPP8kyqRtZLBY899xz2Lx5M8xmM8aPH4/XXnsNycnJHfbeOoM/7t/Zs2fx/PPP4/DhwzCZTOjduzcee+wxzJkzx9mGPnue793zzz+PvLw8VFRUICUlBfPnz8djjz0mS+jpjvfOW9+XLFmCpUuXAnBMy7/88stYv349KisrMWzYMLz66qvIzc11tm/JvTl58iQWL16Mb7/9FlqtFvPnz8eSJUuoVBQhpFMFVWBKCCGEEEK6r+CdWyWEEEIIId0KBaaEEEIIISQgUGBKCCGEEEICAgWmhBBCCCEkIFBgSgghhBBCAgIFpoQQQgghJCBQYEoIIYQQQgICBaaE+NFtt92G3r17o6yszO1cbW0t+vfvjzFjxsBut3dB7wghhJDARoEpIX60atUq1NfX49lnn3U79+KLL+Lq1at4/fXXwfN8F/SOEEIICWwUmBLiR+np6Vi8eDE++ugj7Nmzx3n8+++/xzvvvIOFCxdi2LBhndafurq6TnstQgghpL0oMCXEz373u98hNzcXTzzxBOrr6yGKIp588knEx8fjj3/8o7Nd4/7l/fr1Q1xcHIYMGYK//OUvEEVRdr2VK1di6tSp6NOnD/R6PcaMGYO8vDy3183MzMScOXPw73//GxMnToRer8c777zT4e+XEEII8RemsrJS6upOENLdFBQU4JZbbsHvfvc7JCUlYfHixcjLy8P06dMBONab3nzzzTAYDJg/fz6SkpJQUFCADz74AAsXLsSKFSuc10pPT8eMGTOQlZUFQRDw8ccf4+DBg3jzzTcxd+5cZ7vMzEyEh4ejrKwMCxYsQGpqKtLS0jBx4sTOfvuEEEJIm1BgSkgHWbx4MdavXw+NRoObbroJGzZscJ77v//7P6xevRr79+9H7969ncf//Oc/Y+XKlTh27BhSU1MBOKbjQ0JCnG0kScKtt96KiooK5OfnO49nZmbCYDBg69atmDRpUie8Q0IIIcS/aCqfkA6ybNkyxMTEQJIkvPLKK7Jz27dvx5gxYxAREQGj0ej8N3HiRIiiiIMHDzrbNgalNpsNFRUVKC8vx7hx4/DTTz/BbDbLrtunTx8KSgkhhAQtSg0mpINERESgb9++MBgMiI+Pdx6XJAlnz55FUVER0tPTPT63abmp7du3Y+XKlTh58iQEQZC1q6mpgVqtdj5OS0vz75sghBBCOhEFpoR0MklyrJ6ZMmUKFi1a5LFNnz59AAB79+7F/fffj3HjxmHVqlWIj4+HQqHArl278O6777olSjUNUgkhhJBgQ4EpIZ2MZVmkpKTAZDI1m5i0fft2REREYOvkUHdxAAABUElEQVTWrVAoFM7jX3zxRQf3khBCCOl8tMaUkC4wa9Ys5OfnY+/evW7nqqqqYLPZAAAcxwGAbAq/rKwMH3zwQed0lBBCCOlENGJKSBd48skn8fnnn+Ouu+7CnDlzMGjQIJhMJhQWFmLHjh349ttvodfrccstt2Dt2rWYNWsW7rrrLpSXl2PdunVITEyE0Wjs6rdBCCGE+BUFpoR0gbCwMOzevRsrV67E9u3b8Y9//APh4eHo27cvnnnmGURFRQFwrEN9/fXX8frrr2Pp0qVITk7Gb3/7WygUCjz55JNd/C4IIYQQ/6I6poQQQgghJCDQGlNCCCGEEBIQKDAlhBBCCCEBgQJTQgghhBASECgwJYQQQgghAYECU0IIIYQQEhAoMCWEEEIIIQGBAlNCCCGEEBIQKDAlhBBCCCEBgQJTQgghhBASECgwJYQQQgghAeH/A/wIsbWvXhKIAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7fe674c29a90>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "ak = murder_rates.where('State', 'Alaska').drop('State', 'Population').relabeled(1, 'Murder rate in Alaska')\n",
    "mn = murder_rates.where('State', 'Minnesota').drop('State', 'Population').relabeled(1, 'Murder rate in Minnesota')\n",
    "\n",
    "\n",
    "ak_mn = ak.join('Year', mn)\n",
    "ak_mn.plot('Year')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/NkM3Gp\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q1_2\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now what about the murder rates of other states? Say, for example, California and New York? Plot the murder rates of different pairs of states."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4cb2700af64545059824c2b91ac5e0f1",
       "version_major": 2,
       "version_minor": 0
      },
      "text/html": [
       "<p>Failed to display Jupyter Widget of type <code>interactive</code>.</p>\n",
       "<p>\n",
       "  If you're reading this message in the Jupyter Notebook or JupyterLab Notebook, it may mean\n",
       "  that the widgets JavaScript is still loading. If this message persists, it\n",
       "  likely means that the widgets JavaScript library is either not installed or\n",
       "  not enabled. See the <a href=\"https://ipywidgets.readthedocs.io/en/stable/user_install.html\">Jupyter\n",
       "  Widgets Documentation</a> for setup instructions.\n",
       "</p>\n",
       "<p>\n",
       "  If you're reading this message in another frontend (for example, a static\n",
       "  rendering on GitHub or <a href=\"https://nbviewer.jupyter.org/\">NBViewer</a>),\n",
       "  it may mean that your frontend doesn't currently support widgets.\n",
       "</p>\n"
      ],
      "text/plain": [
       "interactive(children=(Dropdown(description='state1', index=4, options=('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachuse', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Isla', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'), value='California'), Dropdown(description='state2', index=31, options=('Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachuse', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Isla', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'), value='New York'), Output()), _dom_classes=('widget-interact',))"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from ipywidgets import interact, interactive, fixed, interact_manual\n",
    "import ipywidgets as widgets\n",
    "\n",
    "def state(state1, state2):\n",
    "    state1_table = murder_rates.where('State', state1).drop('State', 'Population').relabeled(1, 'Murder rate in {}'.format(state1))\n",
    "    state2_table = murder_rates.where('State', state2).drop('State', 'Population').relabeled(1, 'Murder rate in {}'.format(state2))\n",
    "    s1_s2 = state1_table.join('Year', state2_table)\n",
    "    s1_s2.plot('Year')\n",
    "    plots.show()\n",
    "\n",
    "states_array = murder_rates.group('State').column('State')\n",
    "\n",
    "_ = interact(state,\n",
    "             state1=widgets.Dropdown(options=list(states_array),value='California'),\n",
    "             state2=widgets.Dropdown(options=list(states_array),value='New York')\n",
    "            )"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 1.3.** Implement the function `most_murderous`, which takes a year (an integer) as its argument. It does two things:\n",
    "1. It draws a horizontal bar chart of the 5 states that had the highest murder rate in that year.\n",
    "2. It returns an array of the names of these states in order of *increasing* murder rate.\n",
    "\n",
    "Assume that the argument is a year in `murder_rates`. You do not need to check that it is."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "manual_problem_id": "murder_rates_3"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['California', 'Mississippi', 'Texas', 'New York', 'Louisiana'], \n",
       "      dtype='<U14')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nRK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f53180a5b38>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def most_murderous(year):\n",
    "    # Assign most to a table of the most murderous states this year in ascending order.\n",
    "    most = murder_rates.where('Year', year).sort('Murder Rate', descending=True).take(np.arange(5)).sort('Murder Rate')\n",
    "    most.barh('State', 'Murder Rate')\n",
    "    return most.column('State')\n",
    "\n",
    "most_murderous(1990) # California, Mississippi, ..., "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/W6Y31v\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\nFHr16sWhQ4do3rw5J06coFu3brz//vv8+OOPTJo0iXnz5rFs2bJU9ZvavszNzfHz8yMsLIzly5dz\n7NgxRo0alaZ9iIjImyvDThlkz56dRYsWMXjwYD7//HMqVKhAzZo1adWqleG+BhcuXOCbb77h66+/\npk6dOgAsXboUd3d3Nm3aRJcuXVLcT86cOcmXLx9mZmZGN096plGjRvj6+gLQp08fli5dyr59+6hR\no0ai80VERBg+8TE5LVu2NHq8aNEi7O3tOXbsGB4eHoblvr6+RmMnT55MnTp1GDfun5sAOTs7c+HC\nBebPn0+fPn1S3G9q++rfv79hbMmSJZk6dSqdOnViyZIlmJsnngvf7z4+1fsXEZGkzRvXLVXjzp07\nZ7IaUrq5U4ZeQ9CyZUuaNGlCWFgYhw8fZs+ePfj7+/Pf//6X4cOHEx4ejrm5udGLc/78+Slbtixn\nzpxJlxrKlStn9NjOzo7IyMgkxyd36P7f/vjjDz7++GOOHj3KnTt3iIuLIy4ujqtXrxqNe/5IQ3h4\neIJ3aHh4eDBz5kzu379vOAKRkpT62rdvH/PmzePs2bPcv3+f2NhYYmJiuHnzZpIfaZ0nT55U7ft1\n9vDhQ/X5hskqvWaVPuHN6DU1d1rM7DsyZtgpg2dy5cpFw4YNGT16NN9//z2dO3dmxowZCQ6tP+/Z\nWx3Nzc0TvEin5ZMWn7+Iz8zMLNkXfScnJ86ePZvivD4+Pty+fZtPP/2U3bt3s3//frJnz56gr7T8\np37+g6WSk1xfly9fxsfHh7feeotVq1axd+9e/P39AVL8vouISNaQ4YHgea6urjx9+pTo6GhcXV2J\ni4szOvd9//59fv/9d1xdXQEoXLgwjx494v79+4Yxv/76q9GcOXPmTLdbKXt7e7N3716OHz+eYF1c\nXBz379/n7t27nD17lmHDhtGgQQNcXV3566+/UhVUXF1d+emnn4yWhYWFUbx4caysrNKlh+PHjxMT\nE4Ofnx81atTA2dmZ69evp8vcIiLyZsiwQHD37l1atGjBhg0b+O2337h48SLbtm1jwYIF1K9fn3z5\n8uHk5ESzZs0YOnQoP/74I6dOncLX1xcrKyu8vb0BqFatGnny5GHq1KlERESwfft2VqxYYbQvBwcH\noqOjCQkJ4c6dO0afx5BW/fr1o1atWrRq1YolS5Zw8uRJLl68SHBwME2bNuWXX37B2tqaQoUKsXr1\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5311fb6208>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\nFHr16sWhQ4do3rw5J06coFu3brz//vv8+OOPTJo0iXnz5rFs2bJU9ZvavszNzfHz8yMsLIzly5dz\n7NgxRo0alaZ9iIjImyvDThlkz56dRYsWMXjwYD7//HMqVKhAzZo1adWqleG+BhcuXOCbb77h66+/\npk6dOgAsXboUd3d3Nm3aRJcuXVLcT86cOcmXLx9mZmZGN096plGjRvj6+gLQp08fli5dyr59+6hR\no0ai80VERBg+8TE5LVu2NHq8aNEi7O3tOXbsGB4eHoblvr6+RmMnT55MnTp1GDfun5sAOTs7c+HC\nBebPn0+fPn1S3G9q++rfv79hbMmSJZk6dSqdOnViyZIlmJsnngvf7z4+1fsXEZGkzRvXLVXjzp07\nZ7IaUrq5U4ZeQ9CyZUuaNGlCWFgYhw8fZs+ePfj7+/Pf//6X4cOHEx4ejrm5udGLc/78+Slbtixn\nzpxJlxrKlStn9NjOzo7IyMgkxyd36P7f/vjjDz7++GOOHj3KnTt3iIuLIy4ujqtXrxqNe/5IQ3h4\neIJ3aHh4eDBz5kzu379vOAKRkpT62rdvH/PmzePs2bPcv3+f2NhYYmJiuHnzZpIfaZ0nT55U7ft1\n9vDhQ/X5hskqvWaVPuHN6DU1d1rM7DsyZtgpg2dy5cpFw4YNGT16NN9//z2dO3dmxowZCQ6tP+/Z\nWx3Nzc0TvEin5ZMWn7+Iz8zMLNkXfScnJ86ePZvivD4+Pty+fZtPP/2U3bt3s3//frJnz56gr7T8\np37+g6WSk1xfly9fxsfHh7feeotVq1axd+9e/P39AVL8vouISNaQ4YHgea6urjx9+pTo6GhcXV2J\ni4szOvd9//59fv/9d1xdXQEoXLgwjx494v79+4Yxv/76q9GcOXPmTLdbKXt7e7N3716OHz+eYF1c\nXBz379/n7t27nD17lmHDhtGgQQNcXV3566+/UhVUXF1d+emnn4yWhYWFUbx4caysrNKlh+PHjxMT\nE4Ofnx81atTA2dmZ69evp8vcIiLyZsiwQHD37l1atGjBhg0b+O2337h48SLbtm1jwYIF1K9fn3z5\n8uHk5ESzZs0YOnQoP/74I6dOncLX1xcrKyu8vb0BqFatGnny5GHq1KlERESwfft2VqxYYbQvBwcH\noqOjCQkJ4c6dO0afx5BW/fr1o1atWrRq1YolS5Zw8uRJLl68SHBwME2bNuWXX37B2tqaQoUKsXr1\n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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5311f38c50>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "_ = ok.grade(\"q1_3\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 1.4.** How many more people were murdered in California in 1988 than in 1975? Assign `ca_change` to the answer.\n",
    "\n",
    "Recall the formula given at the beginning of the project:\n",
    "\n",
    "$$\\text{murder rate for state X in year Y} = \\frac{\\text{number of murders in state X in year Y}}{\\text{population in state X in year Y}}*100000$$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "726.0"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ca = murder_rates.where('State', are.equal_to('California'))\n",
    "pop_1988 = ca.where('Year', 1988).column('Population').item(0)\n",
    "pop_1975 = ca.where('Year', 1975).column('Population').item(0)\n",
    "murder_rate_1988 = ca.where('Year', 1988).column('Murder Rate').item(0)\n",
    "murder_rate_1975 = ca.where('Year', 1975).column('Murder Rate').item(0)\n",
    "ca_change = (murder_rate_1988/100000)*pop_1988 - (murder_rate_1975/100000)*pop_1975\n",
    "np.round(ca_change)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/2kYKXN\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade('q1_4')\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Changes in Murder Rates\n",
    "\n",
    "In this section, we'll see how to test null hypotheses such as this: \"For this set of U.S. states, the murder rate was equally likely to go up or down each year.\"\n",
    "\n",
    "Murder rates vary widely across states and years, presumably due to the vast array of differences among states and across US history. Rather than attempting to analyze rates themselves, here we will restrict our analysis to whether or not murder rates increased or decreased over certain time spans. **We will not concern ourselves with how much rates increased or decreased; only the direction of the change** - whether they increased or decreased."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The `np.diff` function takes an array of values and computes the differences between adjacent items of a list or array as such:\n",
    "\n",
    "    [item 1 - item 0 , item 2 - item 1 , item 3 - item 2, ...]\n",
    "\n",
    "Instead, we may wish to compute the difference between items that are two positions apart. For example, given a 5-element array, we may want:\n",
    "\n",
    "    [item 2 - item 0 , item 3 - item 1 , item 4 - item 2]\n",
    "    \n",
    "The `diff_n` function below computes this result."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([  99,  990, 9900])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def diff_n(values, n):\n",
    "    return np.array(values)[n:] - np.array(values)[:-n]\n",
    "\n",
    "diff_n(make_array(1, 10, 100, 1000, 10000), 2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 2.1.** Implement the function `two_year_changes` that takes an array of murder rates for a state, ordered by increasing year. For all two-year periods (e.g., from 1960 to 1962), it computes and returns **the number of increases minus the number of decreases.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Alaska: -5\n",
      "Minnesota: 6\n"
     ]
    }
   ],
   "source": [
    "def two_year_changes(rates):\n",
    "    \"Return the number of increases minus the number of decreases after two years.\"\n",
    "    differences = diff_n(rates, 2)\n",
    "    num_incr = np.count_nonzero(differences > 0 )\n",
    "    num_decr = np.count_nonzero(differences < 0)\n",
    "    return num_incr - num_decr\n",
    "\n",
    "print('Alaska:',    two_year_changes(ak.column('Murder rate in Alaska')))\n",
    "print('Minnesota:', two_year_changes(mn.column('Murder rate in Minnesota')))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/4x1M2n\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q2_1\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can use `two_year_changes` to summarize whether rates are mostly increasing or decreasing over time for some state or group of states. Let's see how it varies across the 50 US states.\n",
    "\n",
    "**Question 2.2.** Assign `changes_by_state` to a table with one row per state that has two columns: the `State` name and the `Murder Rate two_year_changes` statistic computed across all years in our data set for that state."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f53115fcb00>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "changes_by_state = murder_rates.drop(1, 2).group('State', two_year_changes)\n",
    "\n",
    "# a histogram of the two-year changes for the states.\n",
    "# Since there are 50 states, each state contributes 2% to one\n",
    "# bar.\n",
    "changes_by_state.hist(\"Murder Rate two_year_changes\", bins=np.arange(-11, 12, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/82jQgL\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q2_2\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Some states have more increases than decreases (a positive number), while some have more decreases than increases (a negative number). \n",
    "\n",
    "**Question 2.3.** Assign `total_changes` to the total increases minus the total decreases for all two-year periods and all states in our data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total increases minus total decreases, across all states and years: 45\n"
     ]
    }
   ],
   "source": [
    "total_changes = sum(changes_by_state.column(1))\n",
    "print('Total increases minus total decreases, across all states and years:', total_changes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/0RWGMK\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q2_3\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\"More increases than decreases,\" one student exclaims, \"Murder rates tend to go up across two-year periods. What dire times we live in.\"\n",
    "\n",
    "\"Not so fast,\" another student replies, \"Even if murder rates just moved up and down uniformly at random, there would be some difference between the increases and decreases. There were a lot of states and a lot of years, so there were many chances for changes to happen. If state murder rates increase and decrease at random with equal probability, perhaps this difference was simply due to chance!\"\n",
    "\n",
    "**Question 2.4.** Set `num_changes` to the number of distinct two-year periods in the entire data set that could result in a change of a state's murder rate for all states. Include both those periods where a change occurred and the periods where a state's rate happened to stay the same."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2100.0"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "states = murder_rates.sort('State', distinct = True).column('State')\n",
    "total = make_array()\n",
    "\n",
    "for i in np.arange(50):\n",
    "    rates = murder_rates.where('State', states.item(i)).column('Murder Rate')\n",
    "    number_of_chances = len(diff_n(rates, 2))\n",
    "    total = np.append(total, number_of_chances)\n",
    "\n",
    "num_changes = sum(total)\n",
    "num_changes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/pYJynN\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q2_4\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We now have enough information to perform a hypothesis test.\n",
    "\n",
    "> **Null Hypothesis**: State murder rates increase and decrease over two-year periods as if \n",
    "\"increase\" or \"decrease\" were sampled at random from a uniform distribution, like a fair coin flip.\n",
    "\n",
    "Murder rates can be more likely to go up or more likely to go down. Since we observed 45 more increases than decreases for all two year periods in our dataset, we formulate an alternative hypothesis in accordance with our suspicion:\n",
    "\n",
    "> **Alternative Hypothesis**: State murder rates are more likely to increase over two-year periods.\n",
    "\n",
    "If we had observed more decreases than increases, our alternative hypothesis would have been defined accordingly (that state murder rates are more likely to *decrease*). This is typical in statistical testing - we first observe a trend in the data and then run a hypothesis test to confirm or reject that trend.\n",
    "\n",
    "*Technical note*: These changes in murder rates are not random samples from any population. They describe all murders in all states over all recent years. However, we can imagine that history could have been different, and that the observed changes are the values observed in only one possible world: the one that happened to occur. In this sense, we can evaluate whether the observed \"total increases minus total decreases\" is consistent with a hypothesis that increases and decreases are drawn at random from a uniform distribution.\n",
    "\n",
    "*Important requirements for our test statistic:* We want to choose a test statistic for which large positive values are evidence in favor of the alternative hypothesis, and other values are evidence in favor of the null hypothesis. This is because once we've determined the direction of our alternative hypothesis, we only care about the tail in that direction. If, for example, our p-value cutoff was 5%, we'd check to see if our observed test statistic fell within the largest 5% of values in our null hypothesis distribution. \n",
    "\n",
    "Our test statistic should depend only on whether murder rates increased or decreased, not on the size of any change. Thus we choose:\n",
    "\n",
    "> **Test Statistic**: The number of increases minus the number of decreases"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The cell below samples increases and decreases at random from a uniform distribution 100 times. The final column of the resulting table gives the number of increases and decreases that resulted from sampling in this way."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "    <thead>\n",
       "        <tr>\n",
       "            <th>Change</th> <th>Chance</th> <th>Chance sample</th>\n",
       "        </tr>\n",
       "    </thead>\n",
       "    <tbody>\n",
       "        <tr>\n",
       "            <td>Increase</td> <td>0.5   </td> <td>45           </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Decrease</td> <td>0.5   </td> <td>55           </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "</table>"
      ],
      "text/plain": [
       "Change   | Chance | Chance sample\n",
       "Increase | 0.5    | 45\n",
       "Decrease | 0.5    | 55"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "uniform = Table().with_columns(\n",
    "    \"Change\", make_array('Increase', 'Decrease'),\n",
    "    \"Chance\", make_array(0.5,        0.5))\n",
    "uniform.sample_from_distribution('Chance', 100)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 2.5.** Create a simulation below, which samples `num_changes` increases/decreases at random many times and forms an empirical distribution of your test statistic under the null hypothesis. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "for_assignment_type": "student",
    "manual_problem_id": "changes_in_murder_rates_5"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAbUAAAEcCAYAAABAuSr7AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzt3XtYTPkfB/D3VBK5DF1dSqQo91Iuq6yy2tzCLhaLWLmz\nUoi1NhZJrXvL7trktm7JKizLyrq2WKzr0uN+LWRQtG01vz88zc/s1Dg1M01zer+eZ56nOefM93zm\n6zzec875nnMkMplMDiIiIhEw0ncBRERE2sJQIyIi0WCoERGRaDDUiIhINBhqREQkGgw1IiISDYYa\nERGJBkONiIhEg6H2DqmpqfouoUxhf6hinyhjfyhjf6jSZZ8w1IiISDQYakREJBoMNSIiEg2GGhER\niQZDjYiIRIOhRkREosFQIyIi0WCoERGRaJjouwAyLMHz42Bubq5RG1tWzNBSNUREyrinRkREosFQ\nIyIi0WCoERGRaDDUiIhINBhqREQkGgYXaosWLUKnTp1gZ2cHR0dH9O/fH5cvX9Z3WUREVAYYXKgd\nPXoUn332Gfbt24fExESYmJigV69eePbsmb5LIyIiPTO469QSEhKU3n/33Xewt7dHSkoK/P399VQV\nERGVBQa3p/ZfmZmZyM/Ph1Qq1XcpRESkZxKZTCbXdxGaCAwMxPXr13Ho0CEYGxsXugwfp649wfPj\nNG5j8YxAjdsgovLLycmpyHkGd/jxbTNmzEBKSgr27t1bZKAB6jvgXVJTUzX6vBhpepsssfUntxFl\n7A9l7A9VuuwTgw216dOnIyEhAUlJSXBwcNB3OUREVAYYZKhNmzYNO3bsQFJSEpydnfVdDhERlREG\nF2qhoaHYsmULNmzYAKlUirS0NABvDolVqVJFz9UREZE+Gdzox9WrV+Ply5cICAhAo0aNFK/ly5fr\nuzQiItIzg9tTk8lk+i6BiIjKKIPbUyMiIioKQ42IiESDoUZERKLBUCMiItEwuIEiVHL9x8/XdwlE\nRDrFPTUiIhINhhoREYkGQ42IiESDoUZERKLBUCMiItFgqBERkWgw1IiISDQYakREJBoMNSIiEg2G\nGhERiQZDjYiIRIOhRkREosFQIyIi0WCoERGRaDDUiIhINBhqREQkGgw1IiISDcGhFhkZicuXLxc5\n/8qVK4iMjNRKUURERCUhONQWLFiAS5cuFTmfoUZERPqmtcOPmZmZqFChgraaIyIiKjYTdTMvXryI\nCxcuKN6fOHECubm5KsvJZDLExsbCyclJ+xUSEREJpDbUdu3apTikKJFIsGbNGqxZs6bQZaVSKb7/\n/nvtV0hERCSQ2lALDAzEhx9+CLlcDh8fH8yYMQMffPCBynLm5uaoX78+TEzUNkdERKRTalPI1tYW\ntra2AICkpCQ0atQIVlZWpVIYERFRcQneterQoYMu6yAiItJYkaE2btw4SCQSLF26FMbGxhg3btw7\nG5NIJFixYoVWCyQiIhKqyFA7fPgwjIyMkJ+fD2NjYxw+fBgSiURtY++aT0REpEtFhtrbQ/kLe09E\nRFTW8N6PREQkGiUag5+ZmQmZTAa5XK4yz87OTuOiiIiISkJwqGVnZyMyMhLr169HRkZGkcupm0dE\nRKRLgkMtJCQEmzZtQrdu3dCuXTtIpVJd1kVERFRsgkMtKSkJQ4YMwZIlS3RZDxERUYkJHigikUjQ\nokULXdZCRESkEcGh1rVrVxw6dEiHpRAREWlGcKiFhITg5s2bmDhxIk6fPo1Hjx7h8ePHKi8iIiJ9\nEXxOzcPDA8Cbi7A3bNhQ5HIc/UhERPoiONSmTp2q9dtgHTt2DMuXL8dff/2Fhw8fIiYmBoMGDSpy\n+du3bxd6Xi8+Ph6dO3fWam1ERGR4BIfa9OnTtb7yrKwsuLq6YsCAARg9erTgz23fvh1NmzZVvK9R\no4bWayMiIsOj16d6dunSBV26dAEAjB07VvDnatasCRsbG12VRUREBkpwqEVGRr5zGYlEgqlTp2pU\nkBCDBw9GdnY2HB0dMXbsWAQEBOh8nUREVPYJDrUFCxYUOU8ikUAul+s81KpUqYKvv/4abdu2hYmJ\nCfbs2YNhw4Zh5cqV6N+/f5GfS01N1Wi9mn6+rMjKyioT7YilP98mxu+kCfaHMvaHKk36xMnJqch5\ngkPt2bNnKtPy8/Nx584drF69GsePH0d8fHzJKhTIwsICEyZMULxv1aoVMjIysHTpUrWhpq4D3iU1\nNVWjz5cl5ubmGreRlZWlcTti6c8CYtpGtIH9oYz9oUqXfaLRo2eMjIzg4OCAuXPnwtHRsVQOPf6X\nu7s7bty4UerrJSKiskdrz1Nr3749fv31V201J9iFCxc4aISIiABocfTj2bNnYWRUvIzMzMxU7GXl\n5+fj3r17OH/+PGrUqAE7OzvMnj0bf/75JxITEwEAP/30EypUqIDmzZvDyMgIe/fuxerVqxEeHq6t\nr0FERAZMcKht2rSp0OnPnz/H8ePHFXfxL46zZ8+iR48eivcRERGIiIjAgAEDsHLlSjx69Ag3b95U\n+kx0dDTu3r0LY2NjODo6YsWKFWrPpxERUfkhONTUXUdmYWGB4ODgYp9T8/LygkwmK3L+ypUrld4P\nHDgQAwcOLNY6iIio/BAcan/99ZfKNIlEAqlUiqpVq2q1KCIiopIQHGr29va6rIOIiEhjWhv9SERE\npG8MNSIiEg2GGhERiQZDjYiIRIOhRkREoiEo1F69eoWaNWsiOjpa1/UQERGVmKBQq1y5MiwtLVGt\nWjVd10NERFRigg8/9urVCzt27EB+fr4u6yEiIioxwRdfd+/eHUeOHMGHH36IIUOGwMHBAZUqVVJZ\nzt3dXasFEhERCSU41AICAhR/nzp1ChKJRGl+wZOvMzIytFcdERFRMQgOtZiYGF3WQUREpDHBoca7\n4xMRUVlXouvUrl+/jpSUFDx//lzb9RAREZVYsUJt27ZtaNq0KTw8PNC1a1ecO3cOAPD06VO4u7tj\nx44dOimSiIhICMGhtnPnTowcORLOzs6YM2cO5HK5Yp6FhQWcnZ2xefNmnRRJREQkhOBQ++abb/D+\n++8jISGh0PNrrVu3xsWLF7VaHBERUXEIDrVr166he/fuRc63srLCkydPtFIUERFRSQgOtcqVKyMr\nK6vI+Tdv3oSFhYVWiiIiIioJwaHm7e2Nn376CTk5OSrzHj58iLVr18LHx0erxRERERWH4OvUvvzy\nS/j6+uL9999Hr169IJFIsH//fiQnJ2Pt2rUwNjbGtGnTdFkrERGRWoL31BwdHbFv3z7Y2NhgwYIF\nkMvliImJwdKlS9GsWTPs3bsXdnZ2uqyViIhILcF7agDQqFEj7NixAzKZDDdu3EB+fj4cHBxgaWmp\nq/qIiIgEK1aoFZBKpXBzc9N2LURERBopVqjJZDLExMRg3759uHPnDgDA3t4efn5+GDduHKRSqU6K\nJCIiEkLwObUbN26gQ4cOiI6ORm5uLry8vODl5YXc3FxER0fjvffew/Xr13VZKxERkVqC99SmTJmC\nFy9eYOfOnfD29laa9/vvv2Pw4MGYNm0a4uPjtV4kERGREIL31E6cOIHRo0erBBoAdOzYEaNGjcLx\n48e1WhwREVFxCA616tWrqz1nJpVKUb16da0URUREVBKCQ23w4MHYsGEDXr58qTLv+fPn2LBhA4YM\nGaLV4oiIiIpD8Dk1JycnSCQStG7dGgMGDECDBg0AvHlg6ObNm2FlZQUnJyeVZ6r17t1buxUTEREV\nQXCojRw5UvH30qVLVeanp6dj5MiRSs9Zk0gkDDUiIio1gkMtKSlJl3UQERFpTHCodejQQZd1EBER\naUzwQBEiIqKyjqFGRESiwVAjIiLRYKgREZFoMNSIiEg0BIdaZGQkLl++XOT8K1euIDIyUitFERER\nlYTgUFuwYAEuXbpU5HyGGhER6ZvWDj9mZmaiQoUK2mquSMeOHcMnn3wCFxcXSKVSbNy4UefrJCIi\nw6D24uuLFy/iwoULivcnTpxAbm6uynIymQyxsbFwcnLSfoX/kZWVBVdXVwwYMACjR4/W+fqIiMhw\nqA21Xbt2KQ4pSiQSrFmzBmvWrCl0WalUiu+//177Ff5Hly5d0KVLFwDA2LFjdb4+IiIyHGpDLTAw\nEB9++CHkcjl8fHwwY8YMfPDBByrLmZubo379+jAxEXzXLSIiIq1Tm0K2trawtbUF8OaGxo0aNYKV\nlVWpFKZNqampev18WZGVlVUm2hFLf75NjN9JE+wPZewPVZr0ibpTXeXihsaanOtLTU0tlXOFpcHc\n3FzjNrKysjRuRyz9WUBM24g2sD+UsT9U6bJPinW88LfffsP69etx69YtyGQypWenAW/Ou507d06r\nBRIREQklONSWLVuG8PBwWFtbw83NDa6urrqsi4iIqNgEh9qqVavg7e2Nbdu2lcr1aEXJzMzEjRs3\nAAD5+fm4d+8ezp8/jxo1asDOzk5vdRERkf4JvvhaJpMhICBAr4EGAGfPnoW3tze8vb3x+vVrRERE\nwNvbG/Pnz9drXUREpH+C99Tc3d3LxAgeLy8vyGQyfZdBRERlkOA9tejoaOzatQtbt27VZT1EREQl\nJnhPbciQIcjJycHo0aMRHByMWrVqwdjYWGkZiUSClJQUrRdJREQkhOBQs7S0hJWVFRo2bKjLeoiI\niEpMcKjt3r1bl3UQERFpjE++JiIi0ShWqGVkZGDu3Lnw8/ODm5sbTp48qZgeGRmJq1ev6qRIIiIi\nIQQffrx9+zb8/f2RkZEBV1dX3Lp1C69fvwYA1KxZEwkJCXjy5AmioqJ0ViwREZE6gkPtq6++glwu\nR0pKCqpWraoyYKRr164870ZERHol+PDjoUOHEBQUBAcHB0gkEpX59erVw4MHD7RaHBERUXEIDrV/\n/vkHUqm0yPnPnz+HkRHHnRARkf4ITiEXFxccO3asyPm7d+9G8+bNtVIUERFRSQgOtTFjxmDHjh2I\njo7Gs2fPALy5S/61a9cwYsQInD59GuPGjdNZoURERO8ieKBI3759ce/ePcyfP19xR/yPPvoIAGBk\nZITZs2fD399fN1USEREJUKwnXwcHB6Nv375ITEzEjRs3kJ+fj/r166NHjx5wcHDQUYlERETCFCvU\nAKBu3boYO3asLmohIiLSiOBzaikpKVi0aFGR8xcvXqy4wwgREZE+CN5Ti4yMVDuk/+LFizh69Ci2\nb9+ulcKIiIiKS/Ce2vnz5+Hp6VnkfA8PD/z1119aKYqIiKgkBIfaq1evCr2TyNsyMzM1LoiIiKik\nBIdaw4YNcfDgwSLnHzhwAA0aNNBKUURERCUhONSGDBmC/fv3Y+rUqYqLr4E3j52ZMmUKDh48iMGD\nB+ukSCIiIiEEDxQJCgrChQsX8MMPP2D16tWwtrYGAKSnp0Mul2PgwIEYM2aMzgolIiJ6l2Jdp7Zs\n2TLFxde3bt0CADg4OCAgIAAdOnTQRX1ERESCCQq1nJwcnDp1Cra2tvDy8oKXl5eu6yIiIio2QefU\nTExM0KtXL7UDRYiIiPRNUKgZGRnB3t6eQ/aJiKhMEzz6cfTo0YiLi8Pjx491WQ8REVGJCR4o8urV\nK1SuXBlubm7o1q0bHBwcUKlSJaVlJBIJJk6cqPUiiYiIhBAcauHh4Yq/t2zZUugyDDUiItInwaHG\n+zoSEVFZJzjU7O3tdVkHERGRxor9kNDr16/j6NGjePz4Mfr27Yt69eohJycHaWlpsLGxgampqS7q\nJCIieifBoZafn4/g4GCsX78ecrkcEokEHh4eilB77733MGXKFEyYMEGX9RIRERVJ8JD+b775Bhs2\nbMAXX3yB/fv3Qy6XK+ZVqVIFPXr0wK5du3RSJBERkRCCQ23jxo349NNPERISUugjZlxdXXH9+nWt\nFkdERFQcgkPtwYMHcHd3L3J+pUqVeMcRIiLSK8GhZm1tjTt37hQ5/9y5c7Czs9NKUURERCUhONR6\n9uyJ2NhYpUOMEokEALB//35s3rwZvXr10n6FREREAgkOtbCwMNStWxfe3t4ICgqCRCLBokWL0Llz\nZ/Tv3x9NmzbF5MmTdVkrERGRWoJDrVq1avj1118xefJkpKenw8zMDCkpKcjKykJYWBj27Nmjci9I\nIiKi0lSsi6/NzMwQEhKCkJAQXdVDRERUYu8MtezsbOzZswe3b99GzZo14efnB1tb29KojYiIqFjU\nhtrDhw/RtWtX3L59W3GxdeXKlbF582Z4eXlppYDVq1dj2bJlSEtLQ+PGjREREYH27dsXuuyRI0fQ\no0cPleknT56Es7OzVuohIiLDpfac2ty5c3Hnzh2MHTsWW7ZsQUREBMzMzDBt2jStrDwhIQFhYWEI\nCQnB4cOH4enpib59++Lu3btqP5eSkoKrV68qXo6Ojlqph4iIDJvaPbVDhw5hwIABmDt3rmKatbU1\nRowYgfv376NOnToarTwmJgYDBw7E0KFDAQBRUVH47bffEBsbi6+++qrIz1lZWcHCwkKjdRMRkfio\n3VNLS0tDmzZtlKa1bdsWcrkc9+7d02jFOTk5OHfuHHx8fJSm+/j44I8//lD72ffffx+NGjVCz549\ncfjwYY3qICIi8VC7p5aXlwczMzOlaQXvs7OzNVrx06dPkZeXBysrK6XpVlZWSE9PL/Qztra2WLRo\nEdzc3JCTk4MtW7YgICAAu3fvLvI8HACkpqZqVKumny8rsrKyykQ7YunPt4nxO2mC/aGM/aFKkz5x\ncnIqct47Rz/eunULf/75p+L9ixcvFAVVqVJFZXl194fUlJOTk9KX8fT0xJ07d7Bs2TK1oaauA94l\nNTVVo8+XJebm5hq3kZWVpXE7YunPAmLaRrSB/aGM/aFKl33yzlCLiIhARESEyvSpU6cqvS94xlpG\nRoagFVtYWMDY2BiPHz9Wmv748WNYW1sLagN4E6IJCQmClyciIvFSG2oxMTE6W7GpqSlatmyJ5ORk\npXtGJicno2fPnoLbuXDhAmxsbHRRIhERGRi1oTZw4ECdrnzcuHEYNWoU3N3d0aZNG8TGxuLRo0cY\nNmwYAGDUqFEAgO+++w4A8O2338Le3h4uLi7IycnB1q1bsXv3bqxbt06ndRIRkWEo1m2ytK1Pnz7I\nyMhAVFQU0tLS4OLigq1bt8Le3h4AVEZY/vvvv5g1axYePHgAMzMzxfJdunTRR/lERFTG6DXUAGDE\niBEYMWJEofN2796t9P7zzz/H559/XhplERGRARJ8l34iIqKyjqFGRESiwVAjIiLRYKgREZFoMNSI\niEg0GGpERCQaDDUiIhINhhoREYkGQ42IiESDoUZERKLBUCMiItFgqBERkWgw1IiISDQYakREJBoM\nNSIiEg2GGhERiQZDjYiIRIOhRkREosFQIyIi0WCoERGRaDDUiIhINBhqREQkGgw1IiISDYYaERGJ\nBkONiIhEg6FGRESiwVAjIiLRYKgREZFoMNSIiEg0GGpERCQaDDUiIhINhhoREYkGQ42IiESDoUZE\nRKLBUCMiItFgqBERkWgw1IiISDQYakREJBoMNSIiEg2GGhERiQZDjYiIRIOhRkREomGwobZ69Wo0\nb94cNjY26NixI44fP67vkoiISM8MMtQSEhIQFhaGkJAQHD58GJ6enujbty/u3r2r79KIiEiPDDLU\nYmJiMHDgQAwdOhSNGjVCVFQUbGxsEBsbq+/SiIhIjyQymUyu7yKKIycnB7Vq1cKPP/6IXr16KaaH\nhobi8uXL2LNnjx6rIyIifTK4PbWnT58iLy8PVlZWStOtrKyQnp6up6qIiKgsMLhQIyIiKorBhZqF\nhQWMjY3x+PFjpemPHz+GtbW1nqoiIqKywOBCzdTUFC1btkRycrLS9OTkZLRp00ZPVRERUVlgou8C\nSmLcuHEYNWoU3N3d0aZNG8TGxuLRo0cYNmyYvksjIiI9Mrg9NQDo06cPIiIiEBUVBS8vL6SkpGDr\n1q2wt7cvUXtxcXHo3r077O3tIZVKcfv2bZVlZDIZRo4cCXt7e9jb22PkyJGQyWRKy1y6dAldu3aF\nra0tXFxcEBkZCbncoAaXqlVeLng/duwYPvnkE7i4uEAqlWLjxo1K8+VyOSIiItC4cWPY2tqiW7du\nuHLlitIyQrYXQ7Fo0SJ06tQJdnZ2cHR0RP/+/XH58mWlZcpTn/zwww9o37497OzsYGdnhw8++AD7\n9u1TzC9PfVGYRYsWQSqVYsqUKYpppdknBhlqADBixAhcuHAB6enp+P333/Hee++VuK1Xr17Bx8cH\nYWFhatd3/vx5xMfHIz4+HufPn8eoUaMU81+8eIHevXvD2toaBw8exIIFC7B8+XKsWLGixHWVJeXp\ngvesrCy4urpiwYIFqFSpksr8pUuXIiYmBpGRkTh48CCsrKzQu3dvvHz5UrHMu7YXQ3L06FF89tln\n2LdvHxITE2FiYoJevXrh2bNnimXKU5/Url0bs2fPxu+//47k5GR4e3tj0KBBuHjxIoDy1Rf/derU\nKcTFxaFJkyZK00uzTwzuOjVdOnv2LDp16oS//voL9erVU0y/evUq2rRpg71796Jt27YAgBMnTsDf\n3x+nTp2Ck5MTfvzxR4SHh+PatWuK/wijoqIQGxuLy5cvQyKR6OU7aYuvry+aNGmCZcuWKaa5ubkh\nICAAX331lR4r0606depg4cKFGDRoEIA3vzgbN26MoKAghIaGAgBev34NJycnfP311xg2bJig7cWQ\nZWZmwt7eHhs3boS/vz/7BICDgwO++uorBAYGltu+eP78OTp27Ihly5YhMjISrq6uiIqKKvXtw2D3\n1ErTyZMnUaVKFaWBKG3btoW5uTn++OMPxTLt2rVT+mXv6+uLhw8fFno405Dk5OTg3Llz8PHxUZru\n4+Oj+P7lxe3bt5GWlqbUF5UqVUL79u2VtoV3bS+GLDMzE/n5+ZBKpQDKd5/k5eVh+/btyMrKgqen\nZ7nui0mTJiEgIADe3t5K00u7TwxyoEhpS09Ph4WFhdLelkQigaWlpeKC7/T0dNSuXVvpcwUXiKen\np8PBwaHU6tU2XvD+f2lpaQBQaF88fPgQgLDtxZCFhYWhWbNm8PT0BFA+++TSpUvo0qULsrOzYW5u\njg0bNqBJkyaK/4DLU18AwNq1a3Hjxg18//33KvNKe/sQbajNnTsX0dHRapdJSkqCl5dXKVVEZPhm\nzJiBlJQU7N27F8bGxvouR2+cnJxw5MgRvHjxAjt37sSYMWOwa9cufZelF6mpqZgzZw727t2LChUq\n6Lsc8YbamDFj0K9fP7XL1K1bV1Bb1tbWePr0KeRyueKXhFwux5MnTxQXfFtbWxd6QXjBPEPGC97/\nz8bGBsCb725nZ6eY/nZfCNleDNH06dORkJCApKQkpSMP5bFPTE1N0aBBAwBAy5YtcebMGXz77beK\nc0blqS9OnjyJp0+fKs6FAW8Oyx4/fhyxsbFISUkBUHp9ItpzahYWFnB2dlb7qly5sqC2PD09kZmZ\niZMnTyqmnTx5EllZWYpjwJ6enjhx4gSys7MVyyQnJ6NWrVpKg04MES94/7969erBxsZGqS+ys7Nx\n4sQJpW3hXduLoZk2bRq2b9+OxMREODs7K80rr33ytvz8fOTk5JTLvujWrRuOHz+OI0eOKF6tWrXC\nRx99hCNHjqBhw4al2ifGYWFh4Vr5ZgYsLS0NN27cQGpqKpKSkuDj44OsrCyYmpqiUqVKsLS0xOnT\npxEfH49mzZrh/v37CA4Ohpubm2LIqaOjI9asWYMLFy7AyckJJ06cwKxZszBp0iSD3FD/q2rVqoiI\niICtrS3MzMwQFRWF48ePY8WKFahevbq+y9OqzMxM/P3330hLS8P69evh6uqKatWqIScnB9WrV0de\nXh6WLFkCR0dH5OXl4YsvvkBaWhqWLFmCihUrCtpeDEloaCg2b96MuLg41K1bF1lZWcjKygLw5geP\nRCIpV30SHh4OU1NT5Ofn4/79+1i5ciW2bt2K8PBwxfcvL30BAGZmZrCyslJ6bdu2Dfb29hg0aFCp\nbx8c0g8gIiICkZGRKtNjYmIUQ7llMhmmTp2KX375BQDg7++PhQsXKkaAAW9OHoeGhuLMmTOQSqUY\nNmwYpk2bZvDD+QusXr0aS5cuRVpaGlxcXDB//nyNrg8sq44cOYIePXqoTB8wYABWrlwJuVyOBQsW\nIC4uDjKZDO7u7oiOjoarq6tiWSHbi6EoquZp06Zh+vTpAFCu+mTMmDE4cuQI0tPTUa1aNTRp0gQT\nJ06Er68vgPLVF0Xp1q2bYkg/ULp9wlAjIiLREO05NSIiKn8YakREJBoMNSIiEg2GGhERiQZDjYiI\nRIOhRkREosFQI62SSqXvfDVr1kyr69y5cydWrVqllbZyc3MRERGBY8eOlbiN5cuXY8+ePSrTw8PD\nFbeU0kdb+ubs7Izg4GB9l1Fs165dg1Qqxfbt2xXThg8fDg8PDz1WRUUR7b0fST/279+v9P7TTz9F\n06ZNlR7AampqqtV17ty5E2fPnsXo0aM1bis3NxeRkZEwMTEp8YXly5cvh5+fH7p27ao0fcSIEYVe\n1F1abRGVBww10qr//no1NTWFhYUFf9XizQ20hd5EuzTbMkT//PMPKlasqO8yqAzi4UfSq0OHDqFb\nt26oU6cO6tSpg379+uHq1atKy+zduxedO3eGnZ0d6tSpA09PTyxevBjAm8NACQkJuHnzpuLwproA\n/ffffzF79my0aNECNjY2aNCgAfz9/XH69GlkZ2fD1tYWADBv3jxFewXrOnnyJAYNGgRXV1fY2trC\nw8MD8+fPxz///KNo39nZGenp6Vi/fr3i8wWH3Ao7ZLh8+XJ4eHjA1tYWDg4O8PHxwd69e0vU1r//\n/ouoqCh4eHjA2toajo6O6NevH27evFlkfxw4cABSqRSnTp1Smh4bGwupVKp4FlZBPRMmTMCmTZvQ\nunVr1K5dG76+vjh9+rRKu8uXL0fTpk1hY2MDX19flfYL3LhxA8OHD0eDBg1gY2ODjh07Yt++fUrL\nhIeHw8LCApcvX0bPnj1Rp04dtXvlw4cPR6tWrXDmzBl06dIFtWrVgru7O9avX6/SbmGHcHlo0bBx\nT430JjGLxVfgAAAI6klEQVQxEYGBgejevTt++OEH5OXlYfHixejatSuOHTsGW1tbXLt2DYMHD8bH\nH3+M6dOnw8TEBNevX8f9+/cBADNnzkRGRgZSU1MRFxcH4M0NVosSGRmJ1atXY9asWXBxccGLFy9w\n5swZPHv2DBUrVsTu3bvRrVs3DBs2DAMHDgTw/0cU3blzB25ubhg8eDDMzc1x+fJlLFy4EHfv3sXK\nlSsBAFu3bkXv3r3Rtm1bTJ48GYDqwxELrFu3DnPmzEFYWBg8PDzw+vVrXLx4Ec+ePSt2W3K5HJ9+\n+ikOHjyIcePGwcvLC69evcLRo0eRlpaG+vXrF+efpkiHDh3C33//jVmzZsHExARz585Fv379cP78\neVSpUgUA8MMPP+DLL7/E0KFD0bNnT1y7dg2BgYGKmyAXuHXrFnx9fVGnTh1ERkaiZs2a2LJlCwYM\nGIBt27Yp7qVY8P0GDhyIwMBAhIaGvvNZbhkZGRg9ejTGjx+PunXrIi4uDhMmTECjRo0UDzclcWKo\nkV7k5+dj+vTp8PX1xbp16xTTO3TogBYtWmDVqlUIDw/H2bNnkZubq7ibNwB07NhRsXyDBg1Qs2ZN\nmJqaCvp1ferUKfj5+SEoKEgx7e3zVe7u7gCA2rVrq7T38ccfK/6Wy+Vo164dKlWqhODgYCxcuBBV\nq1ZFy5YtUaFCBVhaWr6znlOnTqFVq1YICQlRTPPz81P8XZy29u/fj3379mHJkiUIDAxUTNf2ebdX\nr15h+/btqFatGgCgRo0a8Pf3x8GDB9GzZ0/F3mLXrl2xdOlSAICvry+qV6+OsWPHKrU1b948xQ+J\ngic9+Pr64u7du4iIiFAKtfz8fHz++ecYNmyYoDqfP3+Obdu2KQKsbdu2OHToEOLj4xlqIsfDj6QX\nV65cwf3799GvXz/k5uYqXlWrVoWbmxuOHz8OAGjRogWMjIwwdOhQJCYm4unTpxqt183NDbt378a8\nefPwxx9/4N9//xX8WZlMhi+++AItWrSAtbU1LC0tMXHiROTl5ak9xKeultOnT2P69On4/fff8fr1\n62K3UeDgwYMwMTHBp59+WuI2hGjXrp0i0AAo7rJ+7949AMDt27eRnp6O3r17K32uT58+Kk+rOHDg\nAD788EOYm5srbQM+Pj44c+aM0rMJAaB79+6C65RKpUrhVblyZdSrV09RJ4kXQ4304smTJwCAoKAg\nWFpaKr0OHTqEjIwMAEDjxo0RHx+Pf/75B0FBQXBycoKfn5/iabrFFRYWhpCQECQmJsLPzw+Ojo6Y\nOHEiZDLZOz87cuRIbNy4EWPHjsXPP/+M5ORkzJs3DwCUzqsJNXToUERGRuLEiRPo1asX6tevj6FD\nhyoOrRZHRkYGrK2tYWKi24MvNWrUUHpfsPdc8P0fPXoEQPVp72ZmZqhatarifV5eHp49e4a4uDiV\nf/958+YhPz9f6d/EyMioyEOvQuosqLUk/05kWHj4kfSi4D+duXPnFjp0/u2RbZ06dUKnTp2QnZ2N\nlJQUfP311+jXrx8uXLhQ7AeUVqxYEaGhoQgNDcWjR4/wyy+/YObMmcjJyVF7rduLFy+wf/9+zJkz\nR+mhhWfOnCnW+t9mZGSEoKAgBAUFISMjAwcOHMDMmTMRFBRU6LVp6lhYWCA9PR25ubnFCraCfs7J\nyVGaXvCjorgKBtqkp6crTc/OzsbLly8V742NjVGtWjX4+fmpHJYsYGFhofhbF88krFixInJzc5Gf\nnw8jo///vi/pd6eygXtqpBdNmjRBrVq1cO3aNbRq1Url9fbDAwuYmZnh/fffx/jx4/HixQvFoSRT\nU1OVQ1VC2NraYtiwYWjfvj2uXLmiaEsikai0l52dDblcrhQYcrkcmzZtUmm3JPXUrFkT/fr1Q48e\nPRS1FKctHx8f5ObmYuPGjcVar52dHQAorRNQvd5QqHr16sHa2ho7duxQmp6QkAC5XPnRjZ07d8bF\nixfh6upa6DZQoUKFEtUglJ2dHfLy8nDt2jXFtKdPn2r0Q4X0j3tqpBfGxsZYuHAhAgMD8fr1a/To\n0QM1a9ZEeno6UlJS0LBhQ4wcORKrVq3CmTNn0LlzZ9SuXRtPnz7FN998g7p168LZ2RnAm0OUmzdv\nxrp169C0aVNUqlQJLi4uha73448/hru7O5o3b47q1avj7NmzOHz4sGJvwcjICE5OTtizZw+8vb1R\nrVo11K5dGzY2NmjWrBmWLFkCCwsLVK9eHXFxcYWe42vcuDGOHDmCX3/9FVZWVrC0tFSEx9vGjBkD\nS0tLtG7dGpaWlkhNTcX27duVBkgIbatz587w8/PDlClTcPv2bXTo0AHZ2dk4evQoAgIC0KZNm0L7\nw8HBAa1bt8bChQtRrVo11KhRAz/99BMePHjw7n/EQlSoUAFTpkzBlClTMGnSJPTs2RNXr17FihUr\nYG5urrTsrFmz4Ovri+7du+Ozzz5D3bp1IZPJcOnSJTx69EhxKYWu+Pv7w9zcHOPHj8fUqVPx6tUr\nLF68WOmcIRke7qmR3vTo0QNJSUl49uwZJkyYgI8++gjh4eHIyMhQjEJs3rw5nj9/jvDwcPTp0wfT\npk1Do0aNkJiYqPglP3z4cAQEBGDmzJnw8fHBkCFDilxn+/btceDAAYwbNw59+/bFunXrEBoaipkz\nZyqW+eabb2BsbIy+ffuiU6dO+OmnnwAAa9euhaurK4KDgzF+/HjUq1cPc+bMUVnH7NmzUa9ePQwd\nOhSdOnXCokWLCq2lXbt2OH36NCZPnow+ffpg8eLFGDx4sGLUYHHakkgkWL9+PSZPnoyff/4Z/fv3\nx8SJE3H9+nWV81v/FRsbixYtWiA0NBTjx4+Hk5MTJk6cqPYz6gQFBWHOnDnYv38/BgwYgPj4eKxd\nu1Yl1BwcHJCcnAwnJyeEh4ejd+/eCA0NxR9//IEOHTqUeP1CWVhYYNOmTcjJycHQoUMxf/58TJo0\niaMjDZxEJpPJ370YERFR2cc9NSIiEg2GGhERiQZDjYiIRIOhRkREosFQIyIi0WCoERGRaDDUiIhI\nNBhqREQkGgw1IiISjf8BMmYau6t91TYAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f5311522d68>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def simulate_under_null(num_chances_to_change):\n",
    "    \"\"\"Simulates some number changing several times, with an equal\n",
    "    chance to increase or decrease.  Returns the value of our\n",
    "    test statistic for these simulated changes.\n",
    "    \n",
    "    num_chances_to_change is the number of times the number changes.\n",
    "    \"\"\"\n",
    "    return uniform.sample_from_distribution('Chance', 100).column(2).item(0) - uniform.sample_from_distribution('Chance', 100).column(2).item(1)\n",
    "\n",
    "uniform_samples = make_array()\n",
    "for i in np.arange(5000):\n",
    "    uniform_samples = np.append(uniform_samples, simulate_under_null(num_changes))\n",
    "\n",
    "Table().with_column('Test statistic under null', uniform_samples).hist(0, bins=np.arange(-100, 400+25, 25))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 2.6.** Looking at this histogram, draw a conclusion about whether murder rates basically increase as often as they decrease. (Remember that we're only concerned with the *postive direction* because it supports our alternative hypothesis.)\n",
    "\n",
    "First, set `which_side` to `\"Right\"` or `\"Left\"` depending on which side of the histogram you need to look at to make your conclusion. \n",
    "\n",
    "Then, set `reject_null` to `True` if rates increase more than they decrease, and we can reject the null hypothesis. Set `reject_null` to `False` if they do not systematically increase more than they decrease."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "which_side = 'Right'\n",
    "\n",
    "if (np.count_nonzero(uniform_samples>0) - np.count_nonzero(uniform_samples<0)) > 0:\n",
    "    reject_null = True\n",
    "else: reject_null = False\n",
    "\n",
    "reject_null"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 2\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Backup... 100% complete\n",
      "Backup successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/backups/v2PGwX\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q2_6\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. The death penalty\n",
    "\n",
    "Some US states have the death penalty, and others don't, and laws have changed over time. In addition to changes in murder rates, we will also consider whether the death penalty was in force in each state and each year.\n",
    "\n",
    "Using this information, we would like to investigate how the death penalty affects the murder rate of a state."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 3.1.** Describe this investigation in terms of an experiment. What population are we studying? What is the control group? What is the treatment group? What outcome are we measuring? "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "for_assignment_type": "student",
    "manual_problem_id": "death_penalty_1"
   },
   "source": [
    "\n",
    "\n",
    "- Population: All people living in the United States\n",
    "- Control Group: People living in states during years when the death penalty was not in place. \n",
    "- Treatment Group: People living in states during years the death penalty was in place. \n",
    "- Outcome: The murder rate in a state."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 3.2.** We want to know whether the death penalty *causes* a change in the murder rate.  Why is it not sufficient to compare murder rates in places and times when the death penalty was in force with places and times when it wasn't?"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "manual_problem_id": "death_penalty_2"
   },
   "source": [
    "Murder rates could be overall higher/lower in certain years across all states. Therefore, comparing states with the death penalty in one year vs without the death penalty in another year could give a biased estimate. In addition, some states will have an overall higher/lower murder rate due to population size, location, types of people, etc so we must control for this. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### A Natural Experiment\n",
    "\n",
    "In order to attempt to investigate the causal relationship between the death penalty and murder rates, we're going to take advantage of a *natural experiment*.  A natural experiment happens when something other than experimental design applies a treatment to one group and not to another (control) group, and we have some hope that the treatment and control groups don't have any other systematic differences.\n",
    "\n",
    "Our natural experiment is this: in 1972, a Supreme Court decision called *Furman v. Georgia* banned the death penalty throughout the US.  Suddenly, many states went from having the death penalty to not having the death penalty.\n",
    "\n",
    "As a first step, let's see how murder rates changed before and after the court decision.  We'll define the test as follows:\n",
    "\n",
    "> **Population:** All the states that had the death penalty before the 1972 abolition.  (There is no control group for the states that already lacked the death penalty in 1972, so we must omit them.)  This includes all US states **except** Alaska, Hawaii, Maine, Michigan, Wisconsin, and Minnesota.\n",
    "\n",
    "> **Treatment group:** The states in that population, in the year after 1972.\n",
    "\n",
    "> **Control group:** The states in that population, in the year before 1972.\n",
    "\n",
    "> **Null hypothesis:** Each state's murder rate was equally likely to be higher or lower in the treatment period than in the control period.  (Whether the murder rate increased or decreased in each state was like the flip of a fair coin.)\n",
    "\n",
    "> **Alternative hypothesis:** The murder rate was more likely to increase.\n",
    "\n",
    "Our alternative hypothesis is in keeping with our suspicion that murder rates increase when the death penalty is eliminated.  \n",
    "\n",
    "*Technical Note:* It's not clear that the murder rates were a \"sample\" from any larger population.  Again, it's useful to imagine that our data could have come out differently and to test the null hypothesis that the murder rates were equally likely to move up or down.\n",
    "\n",
    "The `death_penalty` table below describes whether each state allowed the death penalty in 1971."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "    <thead>\n",
       "        <tr>\n",
       "            <th>State</th> <th>Death Penalty</th>\n",
       "        </tr>\n",
       "    </thead>\n",
       "    <tbody>\n",
       "        <tr>\n",
       "            <td>Alabama    </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alaska     </td> <td>False        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Arizona    </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Arkansas   </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>California </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Colorado   </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Connecticut</td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Delaware   </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Florida    </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Georgia    </td> <td>True         </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "</table>\n",
       "<p>... (40 rows omitted)</p"
      ],
      "text/plain": [
       "State       | Death Penalty\n",
       "Alabama     | True\n",
       "Alaska      | False\n",
       "Arizona     | True\n",
       "Arkansas    | True\n",
       "California  | True\n",
       "Colorado    | True\n",
       "Connecticut | True\n",
       "Delaware    | True\n",
       "Florida     | True\n",
       "Georgia     | True\n",
       "... (40 rows omitted)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "non_death_penalty_states = make_array('Alaska', 'Hawaii', 'Maine', 'Michigan', 'Wisconsin', 'Minnesota')\n",
    "def had_death_penalty_in_1971(state):\n",
    "    \"\"\"Returns True if the argument is the name of a state that had the death penalty in 1971.\"\"\"\n",
    "\n",
    "    return state not in non_death_penalty_states\n",
    "\n",
    "states = murder_rates.group('State').select('State')\n",
    "death_penalty = states.with_column('Death Penalty', states.apply(had_death_penalty_in_1971, 0))\n",
    "death_penalty"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "44"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "num_death_penalty_states = death_penalty.where(\"Death Penalty\", are.equal_to(True)).num_rows\n",
    "num_death_penalty_states"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 3.3.** Assign `death_penalty_murder_rates` to a table with the same columns and data as `murder_rates`, but that has only the rows for states that had the death penalty in 1971.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "    <thead>\n",
       "        <tr>\n",
       "            <th>State</th> <th>Year</th> <th>Population</th> <th>Murder Rate</th>\n",
       "        </tr>\n",
       "    </thead>\n",
       "    <tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1960</td> <td>3,266,740 </td> <td>12.4       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1961</td> <td>3,302,000 </td> <td>12.9       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1962</td> <td>3,358,000 </td> <td>9.4        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1963</td> <td>3,347,000 </td> <td>10.2       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1964</td> <td>3,407,000 </td> <td>9.3        </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1965</td> <td>3,462,000 </td> <td>11.4       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1966</td> <td>3,517,000 </td> <td>10.9       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1967</td> <td>3,540,000 </td> <td>11.7       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1968</td> <td>3,566,000 </td> <td>11.8       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>Alabama</td> <td>1969</td> <td>3,531,000 </td> <td>13.7       </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "</table>\n",
       "<p>... (1926 rows omitted)</p"
      ],
      "text/plain": [
       "State   | Year | Population | Murder Rate\n",
       "Alabama | 1960 | 3,266,740  | 12.4\n",
       "Alabama | 1961 | 3,302,000  | 12.9\n",
       "Alabama | 1962 | 3,358,000  | 9.4\n",
       "Alabama | 1963 | 3,347,000  | 10.2\n",
       "Alabama | 1964 | 3,407,000  | 9.3\n",
       "Alabama | 1965 | 3,462,000  | 11.4\n",
       "Alabama | 1966 | 3,517,000  | 10.9\n",
       "Alabama | 1967 | 3,540,000  | 11.7\n",
       "Alabama | 1968 | 3,566,000  | 11.8\n",
       "Alabama | 1969 | 3,531,000  | 13.7\n",
       "... (1926 rows omitted)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "valid_states = death_penalty.where(1, True).column(0)\n",
    "death_penalty_murder_rates = murder_rates.where('State', are.contained_in(valid_states))\n",
    "death_penalty_murder_rates"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The null hypothesis doesn't specify *how* the murder rate changes; it only talks about increasing or decreasing.  So, we will use the same test statistic we defined in section 2."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 3.4.** Assign `test_stat_72` to the value of the test statistic for the years 1971 to 1973 and the states in `death_penalty_murder_rates`. As before, the test statistic is, \"the number of increases minus the number of decreases.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test statistic from 1971 to 1973: 22\n"
     ]
    }
   ],
   "source": [
    "t = death_penalty_murder_rates.where('Year', are.contained_in(make_array(1971, 1973)))\n",
    "increases_and_decreases = make_array()\n",
    "\n",
    "for i in np.arange(len(valid_states)):\n",
    "    state = valid_states.item(i)\n",
    "    change = np.diff(t.where('State', state).column('Murder Rate'))\n",
    "    increases_and_decreases = np.append(increases_and_decreases, change)\n",
    "    \n",
    "increases = np.count_nonzero(increases_and_decreases > 0)\n",
    "decreases = np.count_nonzero(increases_and_decreases < 0)\n",
    "\n",
    "test_stat_72 = increases - decreases \n",
    "        \n",
    "print('Test statistic from 1971 to 1973:', test_stat_72)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Look at the data (or perhaps a random sample) to verify that your answer is correct."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 3.5.**: Draw an empirical histogram of the statistic under the null hypothesis by simulating the test statistic 10,000 times.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {
    "manual_problem_id": "death_penalty_5"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n/T83g/4YLgmtAwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f53114df908>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "samples = make_array()\n",
    "for i in np.arange(10000):\n",
    "    samples = np.append(samples, simulate_under_null(44))\n",
    "Table().with_column('Test statistic under null', samples).hist(bins=np.arange(-4, 28+2, 2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Conclusion\n",
    "**Question 3.6.** Complete the analysis as follows:\n",
    "1. Compute a P-value.\n",
    "2. Draw a conclusion about the null and alternative hypotheses.\n",
    "3. Describe your findings using simple, non-technical language.  Be careful not to claim that the statistical analysis has established more than it really has."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "for_assignment_type": "student",
    "manual_problem_id": "death_penalty_6"
   },
   "source": [
    "**P-value:** 0.0012\n",
    "\n",
    "**Conclusion about the hypotheses:** Our P-value is statistically significant at the 1% level, meaning that the chance that our null hypothesis is true is extrememly unlikeley. We reject our null hypothesis and accept our alternative hypothesis.\n",
    "\n",
    "**Findings:** From our observations we seem to think that the abolition of the death penalty is correlated with an increased murder rate. It could be the case that, without the death penalty enforced in a given state, murders are more likeley to occur."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "for_assignment_type": "student"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.0012"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.count_nonzero(samples >= 22)/10000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Further evidence\n",
    "\n",
    "So far, we have discovered evidence that when executions were outlawed, the murder rate increased in many more states than we would expect from random chance. We have also seen that across all states and all recent years, the murder rate goes up about as much as it goes down over two-year periods. \n",
    "\n",
    "These discoveries seem to support the claim that eliminating the death penalty increases the murder rate. Should we be convinced? Let's conduct some more tests to strengthen our claim.\n",
    "\n",
    "Conducting a test for this data set requires the following steps:\n",
    "\n",
    "1. Select a table containing murder rates for certain states and all years,\n",
    "2. Choose two years and compute the observed value of the test statistic,\n",
    "3. Simulate the test statistic under the null hypothesis that increases and decreases are drawn uniformly at random, then\n",
    "4. Compare the observed difference to the empirical distribution to compute a P-value."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This entire process can be expressed in a single function, called `run_test`.\n",
    "\n",
    "**Question 4.1.** Implement `run_test`, which takes the following arguments:\n",
    "\n",
    "- A table of murder `rates` for certain states, sorted by state and year like `murder_rates`, and\n",
    "- the year when the analysis starts.  (The comparison group is two years later.)\n",
    "\n",
    "It prints out the observed test statistic and returns the P-value for this statistic under the null hypothesis.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test statistic 1971 to 1973 : 22\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.0018"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def run_test(rates, start_year):\n",
    "    \"\"\"Prints out the observed test statistic and returns a P-value for this statistic under the null hypothesis\n",
    "    Parameters\n",
    "    ----------\n",
    "    rates : Table\n",
    "       A table of murder rates for certain states, sorted by state and year like murder_rates\n",
    "    start_year : int\n",
    "       The year when the analysis starts\n",
    "    \"\"\"\n",
    "    num_states = rates.group('State').num_rows\n",
    "    end_year = start_year + 2\n",
    "    table_yrs = rates.where('Year', are.contained_in(make_array(start_year, end_year)))\n",
    "    incr_and_decr = make_array()\n",
    "    \n",
    "    for i in np.arange(num_states):\n",
    "        states = table_yrs.sort('State', distinct = True).column('State')\n",
    "        state = states.item(i)\n",
    "        change = np.diff(table_yrs.where('State', state).column('Murder Rate'))\n",
    "        incr_and_decr = np.append(incr_and_decr, change)\n",
    "        \n",
    "    incr = np.count_nonzero(incr_and_decr > 0)\n",
    "    decr = np.count_nonzero(incr_and_decr < 0)\n",
    "    observed_test_statistic = incr - decr\n",
    "    \n",
    "    samples = make_array()\n",
    "    for i in np.arange(5000):\n",
    "        samples = np.append(samples, simulate_under_null(num_states))\n",
    "        \n",
    "    num_above_test_stat = np.count_nonzero(samples >= observed_test_statistic)\n",
    "    num_below_test_stat = np.count_nonzero(samples <= observed_test_statistic)\n",
    "    \n",
    "    if observed_test_statistic > 0:\n",
    "            p_value = num_above_test_stat/5000\n",
    "    elif observed_test_statistic < 0:\n",
    "            p_value = num_below_test_stat/5000\n",
    "        \n",
    "    print('Test statistic', start_year, 'to', end_year, ':', observed_test_statistic)\n",
    "    return p_value\n",
    "    \n",
    "run_test(death_penalty_murder_rates, 1971)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Submit... 100% complete\n",
      "Submission successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/submissions/Z62v85\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q4_1\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The rest of the states\n",
    "\n",
    "We found a dramatic increase in murder rates for those states affected by the 1972 Supreme Court ruling, but what about the rest of the states? There were six states that had already outlawed execution at the time of the ruling.\n",
    "\n",
    "**Question 4.2.** Create a table called `non_death_penalty_murder_rates` with the same columns as `murder_rates` but only containing rows for the six states without the death penalty in 1971. Perform the same test on this table. **Then**, set reject_null_2 to whether their murder rates were also more likely to increase from 1971 to 1973."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test statistic 1971 to 1973 : 1\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.4586"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "non_death_penalty_murder_rates = murder_rates.where('State', are.contained_in(non_death_penalty_states))\n",
    "run_test(non_death_penalty_murder_rates, 1971)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "reject_null_2 = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Submit... 100% complete\n",
      "Submission successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/submissions/VOL8gv\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q4_2\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The death penalty reinstated\n",
    "\n",
    "In 1976, the Supreme Court repealed its ban on the death penalty in its rulings on [a series of cases including Gregg v. Georgia](https://en.wikipedia.org/wiki/Gregg_v._Georgia), so the death penalty was reinstated where it was previously banned.  This generated a second natural experiment.  To the extent that the death penalty deters murder, reinstating it should decrease murder rates, just as banning it should increase them. Let's see what happened."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Increases minus decreases from 1975 to 1977 (when the death penalty was reinstated) among death penalty states: -18\n",
      "Test statistic 1975 to 1977 : -18\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "0.006"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(\"Increases minus decreases from 1975 to 1977 (when the death penalty was reinstated) among death penalty states:\",\n",
    "      sum(death_penalty_murder_rates.where('Year', are.between_or_equal_to(1975, 1977))\n",
    "                                    .group('State', two_year_changes)\n",
    "                                    .column(\"Murder Rate two_year_changes\")))\n",
    "run_test(death_penalty_murder_rates, 1975)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 4.3.** Now we've analyzed states where the death penalty went away and came back, as well as states where the death penalty was outlawed all along.  What do you conclude from  the results of the tests we have conducted so far? Does all the evidence consistently point toward one conclusion, or is there a contradiction?\n",
    "\n",
    "1) Our results point toward the conclusion that the death penalty moratorium increased murder rates.\n",
    "\n",
    "2) Our results point toward the conclusion that the death penalty moratorium increased murder rates, but we have not accounted for time as a confounding factor.\n",
    "\n",
    "3) Our results don't allow us to make any conclusion about murder rates and death penalties.\n",
    "\n",
    "4) Our results point toward the conclusion that the death penalty moratorium didn't influence murder rates.\n",
    "\n",
    "5) None of these conclusions are valid, or multiple of these conclusions are valid\n",
    "\n",
    "Below, set we_conclude to a single number, corresponding to your answer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "we_conclude = 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Submit... 100% complete\n",
      "Submission successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/submissions/XDNmoW\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade(\"q4_3\")\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Visualization\n",
    "\n",
    "While our analysis appears to support the conclusion that the death penalty deters murder, [a 2006 Stanford Law Review paper](http://users.nber.org/~jwolfers/papers/DeathPenalty%28SLR%29.pdf) argues the opposite: that historical murder rates do **not** provide evidence that the death penalty deters murderers.\n",
    "\n",
    "To understand their argument, we will draw a picture.  In fact, we've gone at this whole analysis rather backward; typically we should draw a picture first and ask precise statistical questions later!\n",
    "\n",
    "What plot should we draw?\n",
    "\n",
    "We know that we want to compare murder rates of states with and without the death penalty.  We know we should focus on the period around the two natural experiments of 1972 and 1976, and we want to understand the evolution of murder rates over time for those groups of states.  It might be useful to look at other time periods, so let's plot them all for good measure."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 5.1.** Create a table called `average_murder_rates` with 1 row for each year in `murder_rates`.  It should have 3 columns:\n",
    "* `Year`, the year,\n",
    "* `Death penalty states`, the average murder rate of the states that had the death penalty in 1971, and\n",
    "* `No death penalty states`, the average murder rate of the other states.\n",
    "\n",
    "`average_murder_rates` should be sorted in increasing order by year. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table border=\"1\" class=\"dataframe\">\n",
       "    <thead>\n",
       "        <tr>\n",
       "            <th>Year</th> <th>Death penalty states</th> <th>No death penalty states</th>\n",
       "        </tr>\n",
       "    </thead>\n",
       "    <tbody>\n",
       "        <tr>\n",
       "            <td>1960</td> <td>5.27955             </td> <td>3.55                   </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1961</td> <td>4.77727             </td> <td>3.68333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1962</td> <td>4.61591             </td> <td>2.33333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1963</td> <td>4.61364             </td> <td>2.75                   </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1964</td> <td>4.71136             </td> <td>3.4                    </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1965</td> <td>4.82727             </td> <td>3.18333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1966</td> <td>5.43182             </td> <td>4.51667                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1967</td> <td>5.875               </td> <td>3.73333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1968</td> <td>6.27045             </td> <td>4.73333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "        <tr>\n",
       "            <td>1969</td> <td>6.50227             </td> <td>4.73333                </td>\n",
       "        </tr>\n",
       "    </tbody>\n",
       "</table>\n",
       "<p>... (34 rows omitted)</p"
      ],
      "text/plain": [
       "Year | Death penalty states | No death penalty states\n",
       "1960 | 5.27955              | 3.55\n",
       "1961 | 4.77727              | 3.68333\n",
       "1962 | 4.61591              | 2.33333\n",
       "1963 | 4.61364              | 2.75\n",
       "1964 | 4.71136              | 3.4\n",
       "1965 | 4.82727              | 3.18333\n",
       "1966 | 5.43182              | 4.51667\n",
       "1967 | 5.875                | 3.73333\n",
       "1968 | 6.27045              | 4.73333\n",
       "1969 | 6.50227              | 4.73333\n",
       "... (34 rows omitted)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "years = murder_rates.sort('Year', distinct = True).column('Year')\n",
    "death_states_avgs = murder_rates.drop('Population').where(0, are.not_contained_in(non_death_penalty_states)).group('Year', np.average).column(2)\n",
    "non_death_states_avgs = murder_rates.drop('Population').where(0, are.contained_in(non_death_penalty_states)).group('Year', np.average).column(2)\n",
    "\n",
    "average_murder_rates = Table().with_columns('Year', years, 'Death penalty states', death_states_avgs, 'No death penalty states', non_death_states_avgs)\n",
    "average_murder_rates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n",
      "Running tests\n",
      "\n",
      "---------------------------------------------------------------------\n",
      "Test summary\n",
      "    Passed: 1\n",
      "    Failed: 0\n",
      "[ooooooooook] 100.0% passed\n",
      "\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_checkpoint();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "IPython.notebook.save_notebook();"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Saving notebook... Saved 'project2.ipynb'.\n",
      "Submit... 100% complete\n",
      "Submission successful for user: sarahtrefler@berkeley.edu\n",
      "URL: https://okpy.org/cal/data8/fa17/project2/submissions/L9xMAW\n",
      "NOTE: this is only a backup. To submit your assignment, use:\n",
      "\tpython3 ok --submit\n",
      "\n"
     ]
    }
   ],
   "source": [
    "_ = ok.grade('q5_1')\n",
    "_ = ok.backup()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 5.2.** Describe in **one short sentence** a high-level takeaway from the line plot below. Are the murder rates in these two groups of states related?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f6d10f735f8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "average_murder_rates.plot('Year')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "manual_problem_id": "visualization_2"
   },
   "source": [
    "Murder rates seem to rise and fall together with a similar trend across all states, both with and without the death penalty, but those states without the death penalty consistently have a lower murder rate during this time period."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Let's bring in another source of information: Canada."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "canada = Table.read_table('canada.csv')\n",
    "murder_rates_with_canada = average_murder_rates.join(\"Year\", canada.select(\"Year\", \"Homicide\").relabeled(\"Homicide\", \"Canada\"))\n",
    "murder_rates_with_canada.plot('Year')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The line plot we generated above is similar to a figure from the [paper](http://users.nber.org/~jwolfers/papers/DeathPenalty%28SLR%29.pdf).\n",
    "\n",
    "<img src=\"paper_plot.png\"/>\n",
    "\n",
    "Canada has not executed a criminal since 1962. Since 1967, the only crime that can be punished by execution in Canada is the murder of on-duty law enforcement personnel. The paper states, \"The most striking finding is that the homicide rate in Canada has moved in\n",
    "virtual lockstep with the rate in the United States.\""
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Question 5.3.** Complete their argument in 2-3 sentences; what features of these plots indicate that the death penalty is not an important factor in determining the murder rate? (If you're stuck, read the [paper](http://users.nber.org/~jwolfers/papers/DeathPenalty%28SLR%29.pdf).)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "manual_problem_id": "visualization_4"
   },
   "source": [
    "We cannot attribute the increases to U.S. homicide rates between 1972-76 to the abolition of the death penalty; prior to the abolishment, murder rates had been steadily increasing for about 9 years (1963-1972) both in the U.S. and Canada. After the death penalty was reestablished in the U.S. until 2001, we see U.S. murder rates slowly decrease overtime with spikes around 1982 and 1992. In Canada, although they abolished the death penalty much earlier in 1967, murder rates also follow the same general trend: a decrease in this time period with spikes around 1982 and 1992. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "manual_problem_id": "visualization_5"
   },
   "source": [
    "**Conclusion**. The plot shows that neither the abolishment nor the institution of the death penalty is the direct cause for changes in murder rates. Evidence from the project also supports this. Firstly, variables such as location and time prevent us from assuming causation. Although we controlled for differences among states by using a natural experiment, our study was limited to the United States, and did not account for trends in other countries. Moreover, trend lines in groups of U.S. states with and without the death penalty follow the same pattern, further emphasizing possible confounding factors that could have an effect on murder rates other than the death penalty. There does not seem to be a causal relationship between increases and decreases to murder rates and whether or not a state has the death penalty. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Murder rates seem to follow a similar trend across differnt countries, unrelated to wether or not the region had the death penalty estabished as law."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Submit the project!\n",
    "_ = ok.submit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# For your convenience, you can run this cell to run all the tests at once!\n",
    "import os\n",
    "print(\"Running all tests...\")\n",
    "_ = [ok.grade(q[:-3]) for q in os.listdir(\"tests\") if q.startswith('q')]\n",
    "print(\"Finished running all tests.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
