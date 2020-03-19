//
//  TestPsxLock.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 12.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class TestPsxLock: XCTestCase {
    
    func testMutex() {
        let mutex = PsxLock()
        let group = DispatchGroup()
        for _ in 0..<10_000 { group.enter() }
        var int = 0
        XCTAssertEqual(int, 0)
        DispatchQueue.concurrentPerform(iterations: 10_000) { _ in
            mutex.perform { int += 1 }
            group.leave()
        }
        group.wait()
        XCTAssertEqual(int, 10_000)
        mutex.free()
    }
    
}
