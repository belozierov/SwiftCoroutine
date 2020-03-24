//
//  TestAtomic.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import Dispatch
@testable import SwiftCoroutine

class TestAtomic: XCTestCase {
    
//    func testUpdate() {
//        measure {
//            let group = DispatchGroup()
//            for _ in 0..<100_000 { group.enter() }
//            var atomic = AtomicInt(wrappedValue: 0)
//            XCTAssertEqual(atomic.wrappedValue, 0)
//            DispatchQueue.concurrentPerform(iterations: 100_000) { _ in
//                while true {
//                    let value = atomic.wrappedValue
//                    if atomic.update(from: value, to: value + 1) { break }
//                }
//                group.leave()
//            }
//            group.wait()
//            XCTAssertEqual(atomic.wrappedValue, 100_000)
//        }
//    }
//    
//    func testUpdate2() {
//        measure {
//            var atomic = AtomicInt(wrappedValue: 0)
//            XCTAssertEqual(atomic.wrappedValue, 0)
//            DispatchQueue.concurrentPerform(iterations: 1_000_000) { _ in
//                let (old, new) = atomic.update { $0 + 1 }
//                XCTAssertEqual(old + 1, new)
//            }
//            XCTAssertEqual(atomic.wrappedValue, 1_000_000)
//        }
//    }
//    
//    func testUpdate5() {
//        measure {
//            var atomic = AtomicInt32(wrappedValue: 0)
//            XCTAssertEqual(atomic.wrappedValue, 0)
//            DispatchQueue.concurrentPerform(iterations: 1_000_000) { _ in
//                let (old, new) = atomic.update { $0 + 1 }
//                XCTAssertEqual(old + 1, new)
//            }
//            XCTAssertEqual(atomic.wrappedValue, 1_000_000)
//        }
//    }
//    
//    func testUpdate3() {
//        measure {
//            var atomic = AtomicInt32(wrappedValue: 0)
//            XCTAssertEqual(atomic.wrappedValue, 0)
//            DispatchQueue.concurrentPerform(iterations: 1_000_000) { _ in
//                atomic.increase()
//            }
//            XCTAssertEqual(atomic.wrappedValue, 1_000_000)
//        }
//    }
//    
//    func testUpdate4() {
//        measure {
//            var atomic = AtomicInt(wrappedValue: 0)
//            XCTAssertEqual(atomic.wrappedValue, 0)
//            DispatchQueue.concurrentPerform(iterations: 1_000_000) { _ in
//                atomic.increase()
//            }
//            XCTAssertEqual(atomic.wrappedValue, 1_000_000)
//        }
//    }
//    
//    func testUpdate7() {
//        measure {
//            var atomic = AtomicInt(wrappedValue: 0)
//            DispatchQueue.concurrentPerform(iterations: 1_000_000) { i in
//                atomic.update { $0 + (i % 2 == 0 ? -1 : 1) }
//            }
//            XCTAssertEqual(atomic.wrappedValue, 0)
//        }
//    }
    
}
