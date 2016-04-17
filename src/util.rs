//
// RustQ - library for pricing financial derivatives written in Rust
// Copyright (c) 2016 by Albert Pang <albert.pang@me.com>
// All rights reserved.
//
// This file is a part of RustQ
//
// RustQ is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// RustQ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
pub fn equal_within(x: f64, y:f64, e: f64) -> bool {
    (x-y).abs() < e
}
