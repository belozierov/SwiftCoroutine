//
//  StorageTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class StorageTests: XCTestCase {
    
    func testPerformance() {
        var storage = Storage<Int>()
        measure {
            DispatchQueue.concurrentPerform(iterations: 100_000) { index in
                storage.remove(storage.append(index))
            }
        }
        storage.free()
    }
    
    func testConcurency() {
        var storage = Storage<Int>()
        DispatchQueue.concurrentPerform(iterations: 100_000) { index in
            let key = storage.append(index)
            var hasValue = false
            storage.forEach { if $0 == index { hasValue = true } }
            XCTAssertTrue(hasValue)
            XCTAssertEqual(storage.remove(key), index)
            XCTAssertNil(storage.remove(key))
        }
        XCTAssertTrue(storage.isEmpty)
        storage.free()
    }
    
    func testAddRemove() {
        var array = [Int]()
        var storage = Storage<Int>()
        storage.append(1)
        let key = storage.append(2)
        storage.append(3)
        XCTAssertEqual(storage.remove(key), 2)
        XCTAssertFalse(storage.isEmpty)
        storage.forEach { array.append($0) }
        XCTAssertEqual(array.count, 2)
        XCTAssertTrue(array.contains(1))
        XCTAssertTrue(array.contains(3))
        storage.removeAll()
        XCTAssertTrue(storage.isEmpty)
        array.removeAll()
        storage.forEach { array.append($0) }
        XCTAssertTrue(array.isEmpty)
        storage.free()
    }
    
}
